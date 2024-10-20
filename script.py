# pip install pexpect
import json
import re
from datetime import datetime  # Import datetime for timestamp generation
from pathlib import Path
import os
import uuid
import base64
import gc
import gradio as gr
import torch
from PIL import Image
#from deepseek_vl.models import VLChatProcessor
#from deepseek_vl.utils.io import load_pil_images as load_pil_images_for_deepseek
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, AutoProcessor, \
    PaliGemmaForConditionalGeneration

model_names = ["phiVision", "DeepSeek", "paligemma", "paligemma_cpu", "minicpm_llama3", "bunny"]


# Load configuration settings from a JSON file
def load_config():
    with open(Path(__file__).parent / "config.json", "r") as config_file:
        # Read and parse the JSON configuration file
        config = json.load(config_file)
    return config


# Load the configuration settings at the module level
config = load_config()

# Define the model ID for PhiVision using the configuration setting
phiVision_model_id = config["phiVision_model_id"]

# Define the model ID for PaliGemma using the configuration setting
paligemma_model_id = config["paligemma_model_id"]

minicpm_llama3_model_id = config["minicpm_llama3_model_id"]

bunny_model_id = config["bunny_model_id"]

cuda_device = config["cuda_visible_devices"]

selected_vision_model = config["default_vision_model"]

# Global variable to store the file path of the selected image
selected_image_path = None

# Define the directory where the image files will be saved
# This path is loaded from the configuration file
image_history_dir = Path(config["image_history_dir"])
# Ensure the directory exists, creating it if necessary
image_history_dir.mkdir(parents=True, exist_ok=True)

global phiVision_model, phiVision_processor
global minicpm_llama_model, minicpm_llama_tokenizer
global paligemma_model, paligemma_processor
global paligemma_cpu_model, paligemma_cpu_processor
global deepseek_processor, deepseek_tokenizer, deepseek_gpt
global bunny_model, bunny_tokenizer


# Function to generate a timestamped filename for saved images
def get_timestamped_filename(extension=".png"):
    # Generate a timestamp in the format YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Return a formatted file name with the timestamp
    return f"image_{timestamp}{extension}"


# Function to modify the user input before it is processed by the LLM
def input_modifier(user_input, state):
    global selected_image_path
    # Check if an image has been selected and stored in the global variable
    if selected_image_path:
        # Construct the message with the "File location" trigger phrase and the image file path
        image_info = f"File location: {selected_image_path}"
        # Combine the user input with the image information, separated by a newline for clarity
        combined_input = f"{user_input}\n\n{image_info}"
        # Reset the selected image path to None after processing
        selected_image_path = None
        return combined_input
    # If no image is selected, return the user input as is
    return user_input


# Function to handle image upload and store the file path
def save_and_print_image_path(image):
    global selected_image_path
    if image is not None:
        # Generate a unique timestamped filename for the image
        file_name = get_timestamped_filename()
        # Construct the full file path for the image
        file_path = image_history_dir / file_name
        # Save the uploaded image to the specified directory
        image.save(file_path, format='PNG')
        print(f"Image selected: {file_path}")
        # Update the global variable with the new image file path
        selected_image_path = file_path
    else:
        # Print a message if no image is selected
        print("No image selected yet.")


# Function to create the Gradio UI components with the new direct vision model interaction elements
def ui():
    # Create an image input component for the Gradio UI
    image_input = gr.Image(label="Select an Image", type="pil", source="upload")
    # Set up the event that occurs when an image is uploaded
    image_input.change(
        fn=save_and_print_image_path,
        inputs=image_input,
        outputs=None
    )

    # Create a text field for user input to the vision model
    user_input_to_vision_model = gr.Textbox(label="Type your question or prompt for the vision model", lines=2,
                                            placeholder="Enter your text here...")

    # Create a button to trigger the vision model processing
    vision_model_button = gr.Button(value="Ask Vision Model")

    # Create a text field to display the vision model's response
    vision_model_response_output = gr.Textbox(
        label="Vision Model Response",
        lines=5,
        placeholder="Response will appear here..."
    )

    # Add radio buttons for vision model selection
    vision_model_selection = gr.Radio(
        choices=model_names + ["got_ocr", "aria"],
        value=config["default_vision_model"],
        label="Select Vision Model"
    )

    # Add an event handler for the radio button selection
    vision_model_selection.change(
        fn=update_vision_model,
        inputs=vision_model_selection,
        outputs=None
    )

    cuda_devices_input = gr.Textbox(
        value=cuda_device,
        label="CUDA Device ID",
        max_lines=1
    )

    cuda_devices_input.change(
        fn=update_cuda_device,
        inputs=cuda_devices_input,
        outputs=None
    )

    # Set up the event that occurs when the vision model button is clicked
    vision_model_button.click(
        fn=process_with_vision_model,
        inputs=[user_input_to_vision_model, image_input, vision_model_selection],
        outputs=vision_model_response_output
    )

    # Return a column containing the UI components
    return gr.Column(
        [
            image_input,
            user_input_to_vision_model,
            vision_model_button,
            vision_model_response_output,
            vision_model_selection
        ]
    )



# Function to load the PaliGemma CPU model and processor
def load_paligemma_cpu_model():
    global paligemma_cpu_model, paligemma_cpu_processor
    paligemma_cpu_model = PaliGemmaForConditionalGeneration.from_pretrained(
        paligemma_model_id,
    ).eval()
    paligemma_cpu_processor = AutoProcessor.from_pretrained(paligemma_model_id)
    print("PaliGemma CPU model loaded on-demand.")


# Function to unload the PaliGemma CPU model and processor
def unload_paligemma_cpu_model():
    global paligemma_cpu_model, paligemma_cpu_processor
    if paligemma_cpu_model is not None:
        # Delete the model and processor instances
        del paligemma_cpu_model
        del paligemma_cpu_processor
        print("PaliGemma CPU model unloaded.")

    # Global variable to store the selected vision model


# Function to update the selected vision model and load the corresponding model
def update_vision_model(model_name):
    global selected_vision_model
    selected_vision_model = model_name
    return model_name


def update_cuda_device(device):
    global cuda_device
    print(f"Cuda device set to index = {device}")
    cuda_device = int(device)
    return cuda_device


# Main entry point for the Gradio interface
if __name__ == "__main__":
    # Launch the Gradio interface with the specified UI components
    gr.Interface(
        fn=ui,
        inputs=None,
        outputs="ui"
    ).launch()

# Define a regular expression pattern to match the "File location" trigger phrase
file_location_pattern = re.compile(r"File location: (.+)$", re.MULTILINE)
# Define a regular expression pattern to match and remove unwanted initial text
unwanted_initial_text_pattern = re.compile(r".*?prod\(dim=0\)\r\n", re.DOTALL)

# Define a regular expression pattern to match and remove unwanted prefixes from DeepSeek responses
# This pattern should be updated to match the new, platform-independent file path format
unwanted_prefix_pattern = re.compile(
    r"^" + re.escape(str(image_history_dir)) + r"[^ ]+\.png\s*Assistant: ",
    re.MULTILINE
)


# Function to load the PhiVision model and processor
def load_phi_vision_model():
    global phiVision_model, phiVision_processor, cuda_device
    # Load the PhiVision model and processor on-demand, specifying the device map
    phiVision_model = AutoModelForCausalLM.from_pretrained(
        phiVision_model_id,
        device_map={"": cuda_device},  # Use the specified CUDA device(s)
        trust_remote_code=True,
        torch_dtype="auto"
    )
    phiVision_processor = AutoProcessor.from_pretrained(
        phiVision_model_id,
        trust_remote_code=True
    )
    print("PhiVision model loaded on-demand.")


# Function to unload the PhiVision model and processor
def unload_phi_vision_model():
    global phiVision_model, phiVision_processor
    if phiVision_model is not None:
        # Delete the model and processor instances
        del phiVision_model
        del phiVision_processor
        # Clear the CUDA cache to free up VRAM
        torch.cuda.empty_cache()
        print("PhiVision model unloaded.")


# Function to load the MiniCPM-Llama3 model and tokenizer
def load_minicpm_llama_model():
    global minicpm_llama_model, minicpm_llama_tokenizer, cuda_device
    if "int4" in minicpm_llama3_model_id:
        # Load the 4-bit quantized model and tokenizer
        minicpm_llama_model = AutoModel.from_pretrained(
            minicpm_llama3_model_id,
            device_map={"": cuda_device},
            trust_remote_code=True,
            torch_dtype=torch.float16  # Use float16 as per the example code for 4-bit models
        ).eval()
        minicpm_llama_tokenizer = AutoTokenizer.from_pretrained(
            minicpm_llama3_model_id,
            trust_remote_code=True
        )
        # Print a message indicating that the 4-bit model is loaded
        print("MiniCPM-Llama3 4-bit quantized model loaded on-demand.")
    else:
        # Load the standard model and tokenizer
        minicpm_llama_model = AutoModel.from_pretrained(
            minicpm_llama3_model_id,
            device_map={"": cuda_device},  # Use the specified CUDA device
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        ).eval()
        minicpm_llama_tokenizer = AutoTokenizer.from_pretrained(
            minicpm_llama3_model_id,
            trust_remote_code=True
        )
        print("MiniCPM-Llama3 standard model loaded on-demand.")


def unload_minicpm_llama_model():
    global minicpm_llama_model, minicpm_llama_tokenizer
    if minicpm_llama_model is not None:
        del minicpm_llama_model
        del minicpm_llama_tokenizer
        torch.cuda.empty_cache()
        print("MiniCPM-Llama3 model unloaded.")


# Function to load the PaliGemma model and processor
def load_paligemma_model():
    global paligemma_model, paligemma_processor, cuda_device
    # Load the PaliGemma model and processor on-demand, specifying the device map
    paligemma_model = PaliGemmaForConditionalGeneration.from_pretrained(
        paligemma_model_id,
        device_map={"": cuda_device},  # Use the specified CUDA device(s)
        torch_dtype=torch.bfloat16,
        revision="bfloat16",
    ).eval()
    paligemma_processor = AutoProcessor.from_pretrained(paligemma_model_id)
    print("PaliGemma model loaded on-demand.")


# Function to unload the PaliGemma model and processor
def unload_paligemma_model():
    global paligemma_model, paligemma_processor
    if paligemma_model is not None:
        # Delete the model and processor instances
        del paligemma_model
        del paligemma_processor
        # Clear the CUDA cache to free up VRAM
        torch.cuda.empty_cache()
        print("PaliGemma model unloaded.")


def load_deepseek_model():
    global deepseek_processor, deepseek_tokenizer, deepseek_gpt, cuda_device
    deepseek_processor = VLChatProcessor.from_pretrained(config["deepseek_vl_model_id"])
    deepseek_tokenizer = deepseek_processor.tokenizer
    deepseek_gpt = AutoModelForCausalLM.from_pretrained(
        config["deepseek_vl_model_id"],
        device_map={"": cuda_device},
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(torch.bfloat16).cuda().eval()
    print("DeepSeek model loaded on-demand.")


def unload_deepseek_model():
    global deepseek_processor, deepseek_tokenizer, deepseek_gpt
    if deepseek_processor is not None:
        del deepseek_gpt
        del deepseek_tokenizer
        del deepseek_processor
        print("DeepSeek model unloaded.")
    torch.cuda.empty_cache()


def load_bunny_model():
    global bunny_model, bunny_tokenizer, cuda_device
    torch.cuda.set_device(cuda_device)  # Set default device before loading models
    bunny_model = AutoModelForCausalLM.from_pretrained(
        bunny_model_id,
        torch_dtype=torch.bfloat16,
        device_map={"": cuda_device},  # Use the specified CUDA device
        trust_remote_code=True
    ).to(torch.bfloat16).cuda()
    bunny_tokenizer = AutoTokenizer.from_pretrained(
        bunny_model_id,
        trust_remote_code=True
    )
    print("Bunny model loaded on-demand.")


def unload_bunny_model():
    global bunny_model, bunny_tokenizer
    if bunny_model is not None:
        del bunny_model, bunny_tokenizer
        print("Bunny model unloaded.")
    torch.cuda.empty_cache()


def load_got_ocr_model():
    global got_ocr_model, got_ocr_tokenizer
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    torch.cuda.set_device(0)  # Explicitly set CUDA device

    got_ocr_model = AutoModel.from_pretrained(config["got_ocr_model_id"], trust_remote_code=True, low_cpu_mem_usage=True, device_map='cuda:0', use_safetensors=True).eval().cuda()
    got_ocr_tokenizer = AutoTokenizer.from_pretrained(config["got_ocr_model_id"], trust_remote_code=True)
    print("GOT-OCR model loaded.")

def unload_got_ocr_model():
    global got_ocr_model, got_ocr_tokenizer
    if got_ocr_model is not None:
        del got_ocr_model
        del got_ocr_tokenizer
        torch.cuda.empty_cache()
        print("GOT-OCR model unloaded.")



def process_with_got_ocr_model(image_path, got_mode, fine_grained_mode="", ocr_color="", ocr_box=""):
    load_got_ocr_model()
    try:
        unique_id = str(uuid.uuid4())
        result_path = f"extensions/Lucid_Autonomy/ImageOutputTest/result_{unique_id}.html"

        if got_mode == "plain texts OCR":
            res = got_ocr_model.chat(got_ocr_tokenizer, str(image_path), ocr_type='ocr')
            return res, None
        elif got_mode == "format texts OCR":
            res = got_ocr_model.chat(got_ocr_tokenizer, str(image_path), ocr_type='format', render=True, save_render_file=result_path)
        elif got_mode == "plain multi-crop OCR":
            res = got_ocr_model.chat_crop(got_ocr_tokenizer, str(image_path), ocr_type='ocr')
            return res, None
        elif got_mode == "format multi-crop OCR":
            res = got_ocr_model.chat_crop(got_ocr_tokenizer, str(image_path), ocr_type='format', render=True, save_render_file=result_path)
        elif got_mode == "plain fine-grained OCR":
            res = got_ocr_model.chat(got_ocr_tokenizer, str(image_path), ocr_type='ocr', ocr_box=ocr_box, ocr_color=ocr_color)
            return res, None
        elif got_mode == "format fine-grained OCR":
            res = got_ocr_model.chat(got_ocr_tokenizer, str(image_path), ocr_type='format', ocr_box=ocr_box, ocr_color=ocr_color, render=True, save_render_file=result_path)

        res_markdown = res

        if "format" in got_mode and os.path.exists(result_path):
            with open(result_path, 'r') as f:
                html_content = f.read()
            encoded_html = base64.b64encode(html_content.encode('utf-8')).decode('utf-8')
            iframe_src = f"data:text/html;base64,{encoded_html}"
            iframe = f'<iframe src="{iframe_src}" width="100%" height="600px"></iframe>'
            download_link = f'<a href="data:text/html;base64,{encoded_html}" download="result_{unique_id}.html">Download Full Result</a>'
            return res_markdown, f"{download_link}<br>{iframe}"
        else:
            return res_markdown, None
    except Exception as e:
        return f"Error: {str(e)}", None
    finally:
        unload_got_ocr_model()



aria_model = None
aria_processor = None

def load_aria_model():
    global aria_model, aria_processor
    if aria_model is None:
        print("Loading ARIA model...")
        aria_model = AutoModelForCausalLM.from_pretrained(
            config["aria_model_id"],
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        aria_processor = AutoProcessor.from_pretrained(
            config["aria_model_id"],
            trust_remote_code=True
        )
        print("ARIA model loaded successfully.")
    else:
        print("ARIA model already loaded.")


def unload_aria_model():
    global aria_model, aria_processor
    if aria_model is not None:
        print("Unloading ARIA model...")
        # Move model to CPU before deletion
        aria_model.cpu()
        # Delete the model and processor
        del aria_model
        del aria_processor
        # Set to None to indicate they're unloaded
        aria_model = None
        aria_processor = None
        # Clear CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        # Run garbage collection
        gc.collect()
        print("ARIA model unloaded successfully.")
    else:
        print("ARIA model not loaded, nothing to unload.")

    # Print current GPU memory usage
    if torch.cuda.is_available():
        print(f"Current GPU memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Current GPU memory reserved: {torch.cuda.memory_reserved() / 1e9:.2f} GB")


def process_with_aria_model(image_path, question):
    global aria_model, aria_processor

    if aria_model is None:
        load_aria_model()

    print("Processing image with ARIA model...")
    image = Image.open(image_path).convert("RGB")
    messages = [
        {"role": "user", "content": [
            {"text": None, "type": "image"},
            {"text": question, "type": "text"},
        ]}
    ]

    text = aria_processor.apply_chat_template(messages, add_generation_prompt=True)
    inputs = aria_processor(text=text, images=image, return_tensors="pt")
    inputs["pixel_values"] = inputs["pixel_values"].to(aria_model.dtype)
    inputs = {k: v.to(aria_model.device) for k, v in inputs.items()}

    with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.bfloat16):
        output = aria_model.generate(
            **inputs,
            max_new_tokens=1024,
            stop_strings=["<|im_end|>"],
            tokenizer=aria_processor.tokenizer,
            do_sample=True,
            temperature=0.9,
        )
        output_ids = output[0][inputs["input_ids"].shape[1]:]
        result = aria_processor.decode(output_ids, skip_special_tokens=True)

    print("Image processing complete.")
    return result




# Function to modify the output from the LLM before it is displayed to the user
def output_modifier(output, state, is_chat=False):
    global cuda_device
    # Search for the "File location" trigger phrase in the LLM's output
    file_location_matches = file_location_pattern.findall(output)
    if file_location_matches:
        # Extract the first match (assuming only one file location per output)
        file_path = file_location_matches[0]
        # Extract the questions for the vision model
        questions_section, _ = output.split(f"File location: {file_path}", 1)
        # Remove all newlines from the questions section and replace them with spaces
        questions = " ".join(questions_section.strip().splitlines())

        # Initialize an empty response string
        vision_model_response = ""

        # Check which vision model is currently selected
        if selected_vision_model == "phiVision":
            vision_model_response = generate_phi_vision(file_path, questions)

        elif selected_vision_model == "DeepSeek":
            vision_model_response = generate_deepseek(file_path, questions)

        elif selected_vision_model == "paligemma":
            vision_model_response = generate_paligemma(file_path, questions)

        elif selected_vision_model == "paligemma_cpu":
            vision_model_response = generate_paligemma_cpu(file_path, questions)

        elif selected_vision_model == "minicpm_llama3":
            vision_model_response = generate_minicpm_llama3(file_path, questions)

        elif selected_vision_model == "bunny":
            vision_model_response = generate_bunny(file_path, questions)

        # Append the vision model's responses to the output
        output_with_responses = f"{output}\n\nVision Model Responses:\n{vision_model_response}"
        return output_with_responses
    # If no file location is found, return the output as is
    return output



# Function to generate a response using the MiniCPM-Llama3 model
def generate_minicpm_llama3(file_path, questions):
    global cuda_device
    try:
        load_minicpm_llama_model()
        image = Image.open(file_path).convert("RGB")
        messages = [
            {"role": "user", "content": f"{questions}"}
        ]
        # Define the generation arguments
        generation_args = {
            "max_new_tokens": 896,
            "repetition_penalty": 1.05,
            "num_beams": 3,
            "top_p": 0.8,
            "top_k": 1,
            "temperature": 0.7,
            "sampling": True,
        }
        if "int4" in minicpm_llama3_model_id:
            # Disable streaming for the 4-bit model
            generation_args["stream"] = False
            # Use the model.chat method with streaming enabled
            vision_model_response = ""
            for new_text in minicpm_llama_model.chat(
                    image=image,
                    msgs=messages,
                    tokenizer=minicpm_llama_tokenizer,
                    **generation_args
            ):
                vision_model_response += new_text
                print(new_text, flush=True, end='')
        else:
            minicpm_llama_model.to(f"cuda:{cuda_device}")
            vision_model_response = minicpm_llama_model.chat(
                image=image,
                msgs=messages,
                tokenizer=minicpm_llama_tokenizer,
                **generation_args
            )
            return vision_model_response
    finally:
        unload_minicpm_llama_model()


def process_with_vision_model(user_input, image, selected_model):
    global cuda_device
    # Save the uploaded image to the specified directory with a timestamp
    file_name = get_timestamped_filename()
    file_path = image_history_dir / file_name
    image.save(file_path, format='PNG')
    print(f"Image processed: {file_path}")

    # Initialize an empty response string
    vision_model_response = ""

    # Check which vision model is currently selected
    if selected_model == "phiVision":
        vision_model_response = generate_phi_vision(file_path, user_input)

    elif selected_model == "DeepSeek":
        vision_model_response = generate_deepseek(file_path, user_input)

    elif selected_model == "paligemma":
        vision_model_response = generate_paligemma(file_path, user_input)

    elif selected_model == "paligemma_cpu":
        vision_model_response = generate_paligemma_cpu(file_path, user_input)

    elif selected_model == "minicpm_llama3":
        vision_model_response = generate_minicpm_llama3(file_path, user_input)

    elif selected_model == "bunny":
        vision_model_response = generate_bunny(file_path, user_input)

    elif selected_model == "got_ocr":
        vision_model_response, _ = process_with_got_ocr_model(file_path, got_mode="plain texts OCR")

    elif selected_model == "aria":
        vision_model_response = process_with_aria_model(file_path, user_input)

    # Return the cleaned-up response from the vision model
    return vision_model_response


def generate_paligemma_cpu(file_path, user_input):
    try:
        # Load the PaliGemma CPU model and processor on-demand
        load_paligemma_cpu_model()
        # Load the saved image using PIL
        with Image.open(file_path) as img:
            # Prepare the prompt for the PaliGemma CPU model using the user's question
            prompt = user_input
            model_inputs = paligemma_cpu_processor(text=prompt, images=img, return_tensors="pt")
            input_len = model_inputs["input_ids"].shape[-1]
            # Generate the response using the PaliGemma CPU model
            with torch.inference_mode():
                generation = paligemma_cpu_model.generate(
                    **model_inputs,
                    max_new_tokens=100,
                    do_sample=True  # Set to True for sampling-based generation
                )
                generation = generation[0][input_len:]
                vision_model_response = paligemma_cpu_processor.decode(generation, skip_special_tokens=True)
        # Unload the PaliGemma CPU model and processor after generating the response
        return vision_model_response
    finally:
        unload_paligemma_cpu_model()


def generate_paligemma(file_path, user_input):
    try:
        # Load the PaliGemma model and processor on-demand
        load_paligemma_model()
        # Load the saved image using PIL
        with Image.open(file_path) as img:
            # Prepare the prompt for the PaliGemma model using the user's question
            model_inputs = paligemma_processor(text=user_input, images=img, return_tensors="pt").to(
                f"cuda:{cuda_device}")
            input_len = model_inputs["input_ids"].shape[-1]
            # Generate the response using the PaliGemma model
            with torch.inference_mode():
                generation = paligemma_model.generate(
                    **model_inputs,
                    max_new_tokens=100,
                    do_sample=True  # Set to True for sampling-based generation
                )
                generation = generation[0][input_len:]
                vision_model_response = paligemma_processor.decode(generation, skip_special_tokens=True)
        # Unload the PaliGemma model and processor after generating the response
        return vision_model_response
    finally:
        unload_paligemma_model()


def generate_phi_vision(file_path, user_input):
    global cuda_device
    try:
        # Load the PhiVision model and processor on-demand
        load_phi_vision_model()
        # Load the saved image using PIL
        with Image.open(file_path) as img:
            # Prepare the prompt for the PhiVision model
            messages = [
                {"role": "user", "content": f"<|image_1|>\n{user_input}"}
            ]
            prompt = phiVision_processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            # Extract the CUDA_VISIBLE_DEVICES setting from the config file
            # cuda_devices = config["cuda_visible_devices"]
            # Convert the CUDA_VISIBLE_DEVICES string to an integer (assuming a single device for simplicity)
            # cuda_device_index = int(cuda_devices)

            # Prepare the model inputs and move them to the specified CUDA device
            inputs = phiVision_processor(prompt, [img], return_tensors="pt").to(f"cuda:{cuda_device}")
            # Define the generation arguments
            generation_args = {
                "max_new_tokens": 500,
                "temperature": 1.0,
                "do_sample": True,  # Set to True for sampling-based generation
                # "min_p": 0.95,  # Optionally set a minimum probability threshold
            }
            # Generate the response using the PhiVision model
            generate_ids = phiVision_model.generate(
                **inputs,
                eos_token_id=phiVision_processor.tokenizer.eos_token_id,
                **generation_args
            )
            # Remove input tokens from the generated IDs
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
            # Decode the generated IDs to get the text response
            vision_model_response = phiVision_processor.batch_decode(
                generate_ids,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=False
            )[0]
        # Unload the PhiVision model and processor after generating the response
        return vision_model_response
    finally:
        unload_phi_vision_model()


def generate_deepseek(file_path, user_input):
    try:
        load_deepseek_model()
        conversation = [
            {
                "role": "User",
                "content": f"<image_placeholder>{user_input}",
                "images": [f"{file_path}"]
            }, {
                "role": "Assistant",
                "content": ""
            }
        ]
        print(conversation)
        pil_images = load_pil_images_for_deepseek(conversation)
        prepare_inputs = deepseek_processor(
            conversations=conversation,
            images=pil_images,
            force_batchify=True
        ).to(deepseek_gpt.device)
        input_embeds = deepseek_gpt.prepare_inputs_embeds(**prepare_inputs)
        outputs = deepseek_gpt.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=prepare_inputs.attention_mask,
            pad_token_id=deepseek_tokenizer.eos_token_id,
            bos_token_id=deepseek_tokenizer.bos_token_id,
            eos_token_id=deepseek_tokenizer.eos_token_id,
            max_new_tokens=512,
            do_sample=False,
            use_cache=True
        )
        return deepseek_tokenizer.decode(outputs[0].cpu().tolist(), skip_special_tokens=True)
    finally:
        unload_deepseek_model()


def generate_bunny(file_path, user_input):
    global cuda_device
    try:
        load_bunny_model()
        with Image.open(file_path) as image:
            text = f"A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions. USER: <image>\n{user_input} ASSISTANT:"
            text_chunks = [bunny_tokenizer(chunk).input_ids for chunk in text.split('<image>')]
            input_ids = torch.tensor(text_chunks[0] + [-200] + text_chunks[1][1:], dtype=torch.long).unsqueeze(0).to(
                f"cuda:{cuda_device}")
            image_tensor = bunny_model.process_images(
                [image],
                bunny_model.config
            ).to(
                dtype=bunny_model.dtype,
                device=bunny_model.device
            )
            output_ids = bunny_model.generate(
                input_ids,
                images=image_tensor,
                max_new_tokens=896,
                use_cache=True,
                repetition_penalty=1.0
            )[0]
            return bunny_tokenizer.decode(output_ids[input_ids.shape[1]:], skip_special_tokens=True).strip()
    finally:
        unload_bunny_model()


# Function to modify the chat history before it is used for text generation
def history_modifier(history):
    # Extract all entries from the "internal" history
    internal_entries = history["internal"]
    # Iterate over the "internal" history entries
    for internal_index, internal_entry in enumerate(internal_entries):
        # Extract the text content of the internal entry
        internal_text = internal_entry[1]
        # Search for the "File location" trigger phrase in the internal text
        file_location_matches = file_location_pattern.findall(internal_text)
        if file_location_matches:
            # Iterate over each match found in the "internal" entry
            for file_path in file_location_matches:
                # Construct the full match string including the trigger phrase
                full_match_string = f"File location: {file_path}"
                # Search for the exact same string in the "visible" history
                for visible_entry in history["visible"]:
                    # Extract the text content of the visible entry
                    visible_text = visible_entry[1]
                    # If the "visible" entry contains the full match string
                    if full_match_string in visible_text:
                        # Split the "visible" text at the full match string
                        _, after_match = visible_text.split(full_match_string, 1)
                        # Find the position where the ".png" part ends in the "internal" text
                        png_end_pos = internal_text.find(file_path) + len(file_path)
                        # If the ".png" part is found and there is content after it
                        if png_end_pos < len(internal_text) and internal_text[png_end_pos] == "\n":
                            # Extract the existing content after the ".png" part in the "internal" text
                            _ = internal_text[png_end_pos:]
                            # Replace the existing content after the ".png" part in the "internal" text
                            # with the corresponding content from the "visible" text
                            new_internal_text = internal_text[:png_end_pos] + after_match
                            # Update the "internal" history entry with the new text
                            history["internal"][internal_index][1] = new_internal_text
                        # If there is no content after the ".png" part in the "internal" text,
                        # append the content from the "visible" text directly
                        else:
                            # Append the content after the full match string from the "visible" text
                            history["internal"][internal_index][1] += after_match
    return history
