# pip install pexpect
import json
import os
import re
from datetime import datetime  # Import datetime for timestamp generation
from pathlib import Path

import gradio as gr
import pexpect
import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration


# Load configuration settings from a JSON file
def load_config():
    with open(Path(__file__).parent / "config.json", "r") as config_file:
        # Read and parse the JSON configuration file
        config = json.load(config_file)
    return config


# Load the configuration settings at the module level
config = load_config()

# Define the directory where the image files will be saved
# This path is loaded from the configuration file
image_history_dir = Path(config["image_history_dir"])
# Ensure the directory exists, creating it if necessary
image_history_dir.mkdir(parents=True, exist_ok=True)


# Function to generate a timestamped filename for saved images
def get_timestamped_filename(extension=".png"):
    # Generate a timestamp in the format YYYYMMDD_HHMMSS
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # Return a formatted file name with the timestamp
    return f"image_{timestamp}{extension}"


# Global variable to store the file path of the selected image
selected_image_path = None


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
    vision_model_response_output = gr.Textbox(label="Vision Model Response", lines=5,
                                              placeholder="Response will appear here...")

    # Add radio buttons for vision model selection
    vision_model_selection = gr.Radio(
        choices=["phiVision", "DeepSeek", "paligemma", "paligemma_cpu", "minicpm_llama3"],
        # Added "paligemma_cpu" as a choice
        value=config["default_vision_model"],
        label="Select Vision Model"
    )
    # Add an event handler for the radio button selection
    vision_model_selection.change(
        fn=update_vision_model,
        inputs=vision_model_selection,
        outputs=None
    )

    # Set up the event that occurs when the vision model button is clicked
    vision_model_button.click(
        fn=process_with_vision_model,
        inputs=[user_input_to_vision_model, image_input, vision_model_selection],  # Pass the actual gr.Image component
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


selected_vision_model = "phiVision"


# Function to update the selected vision model and load the corresponding model
def update_vision_model(model_name):
    global selected_vision_model, phiVision_model, phiVision_processor
    selected_vision_model = model_name
    return model_name


# Define the model ID for PhiVision using the configuration setting
phiVision_model_id = config["phiVision_model_id"]

# Define the model ID for PaliGemma using the configuration setting
paligemma_model_id = config["paligemma_model_id"]

minicpm_llama3_model_id = config["minicpm_llama3_model_id"]

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
    global phiVision_model, phiVision_processor
    # Extract the CUDA_VISIBLE_DEVICES setting from the config file
    cuda_devices = config["cuda_visible_devices"]
    # Load the PhiVision model and processor on-demand, specifying the device map
    phiVision_model = AutoModelForCausalLM.from_pretrained(
        phiVision_model_id,
        device_map={"": int(cuda_devices)},  # Use the specified CUDA device(s)
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


def load_minicpm_llama_model():
    global minicpm_llama_model, minicpm_llama_tokenizer
    cuda_devices = config["cuda_visible_devices"]
    minicpm_llama_model = AutoModel.from_pretrained(
        minicpm_llama3_model_id,
        device_map={"": int(cuda_devices)},  # Use the specified CUDA device(s)
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        do_sample=True
    ).eval()
    minicpm_llama_tokenizer = AutoTokenizer.from_pretrained(
        minicpm_llama3_model_id,
        trust_remote_code=True,
        do_sample=True
    )
    print("MiniCPM-Llama3 model loaded on-demand.")


global minicpm_llama_model, minicpm_llama_tokenizer


def unload_minicpm_llama_model():
    global minicpm_llama_model, minicpm_llama_tokenizer
    if minicpm_llama_model is not None:
        del minicpm_llama_model
        del minicpm_llama_tokenizer
        torch.cuda.empty_cache()
        print("MiniCPM-Llama3 model unloaded.")


# Function to load the PaliGemma model and processor
def load_paligemma_model():
    global paligemma_model, paligemma_processor
    # Extract the CUDA_VISIBLE_DEVICES setting from the config file
    cuda_devices = config["cuda_visible_devices"]
    # Load the PaliGemma model and processor on-demand, specifying the device map
    paligemma_model = PaliGemmaForConditionalGeneration.from_pretrained(
        paligemma_model_id,
        device_map={"": int(cuda_devices)},  # Use the specified CUDA device(s)
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


# Function to modify the output from the LLM before it is displayed to the user
def output_modifier(output, state, is_chat=False):
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
            # Load the PhiVision model and processor on-demand
            load_phi_vision_model()

            # Load the image using PIL
            image = Image.open(file_path)

            # Prepare the prompt for the PhiVision model
            messages = [
                {"role": "user", "content": f"<|image_1|>\n{questions}"}
            ]
            prompt = phiVision_processor.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )

            # Extract the CUDA_VISIBLE_DEVICES setting from the config file
            cuda_devices = config["cuda_visible_devices"]
            # Convert the CUDA_VISIBLE_DEVICES string to an integer (assuming a single device for simplicity)
            cuda_device_index = int(cuda_devices)

            # Process the prompt and image to create model inputs
            inputs = phiVision_processor(prompt, [image], return_tensors="pt").to(f"cuda:{cuda_device_index}")

            # Define the generation arguments
            generation_args = {
                "max_new_tokens": 500,
                "temperature": 1.0,
                "do_sample": False,
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
            unload_phi_vision_model()

        elif selected_vision_model == "DeepSeek":
            # Call the function to send the questions and file path to the DeepSeek vision model CLI
            vision_model_responses = send_to_vision_model_cli(file_path, questions)
            # Clean up the DeepSeek vision model's responses by removing the unwanted initial text and prefixes
            cleaned_responses = unwanted_initial_text_pattern.sub("", vision_model_responses)
            cleaned_responses = unwanted_prefix_pattern.sub("", cleaned_responses)

            vision_model_response = cleaned_responses

        elif selected_vision_model == "paligemma":
            # Load the PaliGemma model and processor on-demand
            load_paligemma_model()

            # Load the image using PIL
            image = Image.open(file_path)

            # Prepare the prompt for the PaliGemma model using the user's question
            prompt = questions

            # Extract the CUDA_VISIBLE_DEVICES setting from the config file
            cuda_devices = config["cuda_visible_devices"]
            # Convert the CUDA_VISIBLE_DEVICES string to an integer (assuming a single device for simplicity)
            cuda_device_index = int(cuda_devices)

            model_inputs = paligemma_processor(text=prompt, images=image, return_tensors="pt").to(
                f"cuda:{cuda_device_index}")
            input_len = model_inputs["input_ids"].shape[-1]

            # Generate the response using the PaliGemma model
            with torch.inference_mode():
                generation = paligemma_model.generate(
                    **model_inputs,
                    max_new_tokens=100,
                    do_sample=True
                )
                generation = generation[0][input_len:]
                vision_model_response = paligemma_processor.decode(generation, skip_special_tokens=True)

            # Unload the PaliGemma model and processor after generating the response
            unload_paligemma_model()

        elif selected_vision_model == "paligemma_cpu":
            # Load the PaliGemma CPU model and processor on-demand
            load_paligemma_cpu_model()

            # Load the image using PIL
            image = Image.open(file_path)

            # Prepare the prompt for the PaliGemma CPU model using the user's question
            prompt = questions
            model_inputs = paligemma_cpu_processor(text=prompt, images=image, return_tensors="pt")
            input_len = model_inputs["input_ids"].shape[-1]

            # Generate the response using the PaliGemma CPU model
            with torch.inference_mode():
                generation = paligemma_cpu_model.generate(
                    **model_inputs,
                    max_new_tokens=100,
                    do_sample=False
                )
                generation = generation[0][input_len:]
                vision_model_response = paligemma_cpu_processor.decode(generation, skip_special_tokens=True)

            # Unload the PaliGemma CPU model and processor after generating the response
            unload_paligemma_cpu_model()

        elif selected_vision_model == "minicpm_llama3":
            vision_model_response = generate_minicpm_llama3(file_path, questions)

        # Append the vision model's responses to the output
        output_with_responses = f"{output}\n\nVision Model Responses:\n{vision_model_response}"
        return output_with_responses
    # If no file location is found, return the output as is
    return output


def generate_minicpm_llama3(file_path, questions):
    load_minicpm_llama_model()
    image = Image.open(file_path).convert("RGB")
    messages = [
        {"role": "user", "content": f"{questions}"}
    ]
    cuda_devices = config["cuda_visible_devices"]
    cuda_device_index = int(cuda_devices)
    minicpm_llama_model.to(f"cuda:{cuda_device_index}")
    generation_args = {
        "max_new_tokens": 896,
        "repetition_penalty": 1.05,
        "num_beams": 3,
        "top_p": 0.8,
        "top_k": 1,
        "temperature": 0.7,
        "sampling": True,
    }
    vision_model_response = minicpm_llama_model.chat(
        image=image,
        msgs=messages,
        tokenizer=minicpm_llama_tokenizer,
        **generation_args
    )
    unload_minicpm_llama_model()
    return vision_model_response


# Function to process the user's input text and selected image with the vision model
def process_with_vision_model(user_input, image, selected_model):
    # Save the uploaded image to the specified directory with a timestamp
    file_name = get_timestamped_filename()
    file_path = image_history_dir / file_name
    image.save(file_path, format='PNG')
    print(f"Image processed: {file_path}")

    # Initialize an empty response string
    vision_model_response = ""

    # Check which vision model is currently selected
    if selected_model == "phiVision":
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
            cuda_devices = config["cuda_visible_devices"]
            # Convert the CUDA_VISIBLE_DEVICES string to an integer (assuming a single device for simplicity)
            cuda_device_index = int(cuda_devices)

            # Prepare the model inputs and move them to the specified CUDA device
            inputs = phiVision_processor(prompt, [image], return_tensors="pt").to(f"cuda:{cuda_device_index}")
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
        unload_phi_vision_model()

    elif selected_model == "DeepSeek":
        # Call the function to send the questions and file path to the DeepSeek vision model CLI
        vision_model_responses = send_to_vision_model_cli(file_path, user_input)
        # Clean up the DeepSeek vision model's responses by removing the unwanted initial text and prefixes
        cleaned_responses = unwanted_initial_text_pattern.sub("", vision_model_responses)
        cleaned_responses = unwanted_prefix_pattern.sub("", cleaned_responses)
        vision_model_response = cleaned_responses


    elif selected_model == "paligemma":
        # Load the PaliGemma model and processor on-demand
        load_paligemma_model()
        # Load the saved image using PIL
        with Image.open(file_path) as img:
            # Prepare the prompt for the PaliGemma model using the user's question
            prompt = user_input
            # Extract the CUDA_VISIBLE_DEVICES setting from the config file
            cuda_devices = config["cuda_visible_devices"]
            # Convert the CUDA_VISIBLE_DEVICES string to an integer (assuming a single device for simplicity)
            cuda_device_index = int(cuda_devices)
            model_inputs = paligemma_processor(text=prompt, images=img, return_tensors="pt").to(
                f"cuda:{cuda_device_index}")
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
        unload_paligemma_model()

    elif selected_model == "paligemma_cpu":
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
        unload_paligemma_cpu_model()

    elif selected_model == "minicpm_llama3":
        vision_model_response = generate_minicpm_llama3(file_path, user_input)

    # Return the cleaned-up response from the vision model
    return vision_model_response


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
                        before_match, after_match = visible_text.split(full_match_string, 1)
                        # Find the position where the ".png" part ends in the "internal" text
                        png_end_pos = internal_text.find(file_path) + len(file_path)
                        # If the ".png" part is found and there is content after it
                        if png_end_pos < len(internal_text) and internal_text[png_end_pos] == "\n":
                            # Extract the existing content after the ".png" part in the "internal" text
                            existing_internal_after_png = internal_text[png_end_pos:]
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


# Function to send questions to the vision model CLI and collect the responses
def send_to_vision_model_cli(file_path, questions):
    # Convert file_path to a string for CLI compatibility
    file_path_str = str(file_path)

    # Extract the CUDA_VISIBLE_DEVICES setting from the config file
    cuda_devices = config["cuda_visible_devices"]
    # Ensure that cuda_devices is a string (it could be an integer or a list of integers)
    cuda_devices_str = str(cuda_devices)

    # Define the environment variables for the CLI process
    env = {
        'CUDA_VISIBLE_DEVICES': cuda_devices_str,  # Make sure this is a string
        'PATH': os.environ['PATH']  # Include all necessary paths in the environment
    }
    # Construct the command to run the vision model CLI script with the specified model path
    command = f"{config['python_exec']} {config['cli_script_path']} --model_path {config['model_path']}"
    # Spawn a new process to run the CLI command
    child = pexpect.spawn(command, env=env, encoding='utf-8', timeout=800)
    try:
        # Wait for the initial prompt from the CLI
        child.expect("DeepSeek-VL-Chat is a chatbot that can answer questions based on the given image. Enjoy it!")
        # Send the questions to the CLI, prefixed with an image placeholder
        child.sendline(f"<image_placeholder> {questions}")
        # Wait for the file path prompt and send the image file path as a string
        child.expect(r"\(1/1\) Input the image file path:")
        child.sendline(file_path_str)  # Use the string representation of the file path
        # Wait for the response from the CLI and collect the output
        child.expect("User \[<image_placeholder> indicates an image\]:", timeout=30)
        response = child.before  # Collects all text printed before the matched pattern
        # Close the CLI process
        child.close()
        return response.strip()
    except pexpect.TIMEOUT as e:
        # Handle a timeout error
        child.close()
        return f"Timeout error: {e}"
    except pexpect.EOF as e:
        # Handle an end-of-file error
        child.close()
        return f"EOF error: {e}"
    except Exception as e:
        # Handle any other exceptions that may occur
        child.close()
        return f"Error: {e}"
