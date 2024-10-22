* Added GOT-ORC and Aira (https://github.com/Ucas-HaoranWei/GOT-OCR2.0/  &  https://huggingface.co/rhymes-ai/Aria) to the extension.  If using Aria you will likely need the model ducking extension to unload your llm off the vram to let the Aria model load; it uses about 60GB of vram.

* By Default Deepseek is disabled.  If you want to use Deepseek, you'll need to uncomment the dependencies at the top of the code.

* You can use [Mini-CPM V1.6](https://huggingface.co/openbmb/MiniCPM-V-2_6) [(4-bit too)](https://huggingface.co/openbmb/MiniCPM-V-2_6-int4) instead of 1.5

Video Demo:

There is code in this repo to prevent Alltalk from reading the directory names out loud ([here](https://github.com/RandomInternetPreson/Lucid_Vision/tree/main?tab=readme-ov-file#to-work-with-alltalk)), the video is older and displays Alltalk reading the directory names however. 

https://github.com/RandomInternetPreson/Lucid_Vision/assets/6488699/8879854b-06d8-49ad-836c-13c84eff6ac9

Download the full demo video here:
https://github.com/RandomInternetPreson/Lucid_Vision/blob/main/VideoDemo/Lucid_Vision_demoCompBig.mov

* Update June 2 2024, Lucid_Vision now supports Bunny-v1_1-Llama-3-8B-V, again thanks to https://github.com/justin-luoma :3

* Update May 31 2024, many thanks to https://github.com/justin-luoma, they have made several recent updates to the code, now merged with this repo.
     * Removed the need to communicate with deepseek via the cli, this removes the only extra dependency needed to run Lucid_Vision.
     * Reduced the number of entries needed for the config file.
     * Really cleaned up the code, it is much better organized now.
     * Added the ability to switch which gpu loads the vision model in the UI.
 
* Bonus Update for May 31 2024, WizardLM-2-8x22B and I figured out how to prevent Alltalk from reading the file directory locations!! https://github.com/RandomInternetPreson/Lucid_Vision/tree/main?tab=readme-ov-file#to-work-with-alltalk
  
* Update May 30 2024, Lucid_Vision now supports MiniCPM-Llama3-V-2_5, thanks to https://github.com/justin-luoma.  Additionally WizardLM-2-8x22B added the functionality to load in the MiniCPM 4-bit model.

* Updated script.py and config file, model was not previously loading to the user assigned gpu

Experimental, and I am currently working to improve the code; but it may work for most.

Todo:

* Right now the temp setting and parameters for each model are coded in the .py script, I intend to bring these to the UI
  
[done](https://github.com/RandomInternetPreson/Lucid_Vision/tree/main?tab=readme-ov-file#to-work-with-alltalk)~~* Make it comptable with Alltalk, right now each image directory is printed out and this will be read out loud by Alltalk~~

To accurately proportion credit (original repo creation):

WizardLM-2-8x22B (quantized with exllamaV2 to 8bit precision) = 90% of all the work done in this repo.  That model wrote 100% of all the code and most of the introduction to this repo. 

CommandR+ (quantized with exllamaV2 to 8bit precision) = ~5% of all the work.  CommandR+ contextualized the coding examples and rules for making extensions from Oogabooga's textgen repo extreamly well, and provided a good foundation to develope the code.

RandomInternetPreson = 5% of all the work done.  I came up with the original idea, the original general outline of how the pieces would interact, and provided feedback to the WizardLM model, but I did not write any code.  I'm actually not very good with python yet, with most of my decades of coding being in Matlab.

My goal from the beginning was to write this extension offline without any additional resources, sometimes it was a little frusturating but I soon understood how to get want I needed from the models running locally.

I would say that most of the credit should go to Oobabooga, for without them I would be struggling to even interact with my models.  Please consider supporting them:

https://github.com/sponsors/oobabooga

or 

https://ko-fi.com/oobabooga

I am their top donor on ko-fi (Mr. A) and donate 15$ montly, their software is extreamly important to the opensource community.


# Lucid_Vision Extension for Oobabooga's textgen-webui

Welcome to the Lucid Vision Extension repository! This extension enhances the capabilities of textgen-webui by integrating advanced vision models, allowing users to have contextualized conversations about images with their favorite language models; and allowing direct communciation with vision models.

## Features

* Multi-Model Support: Interact with different vision models, including PhiVision, DeepSeek, MiniCPM-Llama3-V-2_5 (4bit and normal precision), Bunny-v1_1-Llama-3-8B-V, and PaliGemma, with options for both GPU and CPU inference.

* On-Demand Loading: Vision models are loaded into memory only when needed to answer a question, optimizing resource usage.

* Seamless Integration: Easily switch between vision models using a Gradio UI radio button selector.

* Cross-Platform Compatibility: The extension is designed to work on various operating systems, including Unix-based systems and Windows. (not tested in Windows yet, but should probably work?)

* Direct communication with vision models, you do not need to load a LLM to interact with the separate vision models.

## How It Works

The Lucid Vision Extension operates by intercepting and modifying user input and output within the textgen-webui framework. When a user uploads an image and asks a question, the extension appends a special trigger phrase ("File location:") and extracts the associated file path and question.

So if a user enters text into the "send a message" field and has a new picture uploaded into the Lucid_Vision ui, what will happen behind the scenes is the at the user message will be appended with the "File Location: (file location)" Trigger phrase, at which point the LLM will see this and understand that it needs to reply back with questions about the image, and that those questions are being sent to a vison model.

The cool thing is that let's say later in the conversation you want to know something specific about a previous picture, all you need to do is ask your LLM, YOU DO NOT NEED TO REUPLOAD THE PICTURE, the LLM should be able to interact with the extension on its own after you uploaded your first picture.

The selected vision directly interacts with the model's Python API to generate a response. The response is then appended to the chat history, providing the user with detailed insights about the image.

The extension is designed to be efficient with system resources by only loading the vision models into memory when they are actively being used to process a question. After generating a response, the models are immediately unloaded to free up memory and GPU VRAM.


## **How to install and setup:**

1.use the latest version of textgen ~~Install this edited prior commit from oobabooga's textgen https://github.com/RandomInternetPreson/textgen_webui_Lucid_Vision_Testing OR use the latest version of textgen.~~ ~~If using the edited older version, make sure to rename the install folder `text-generation-webui`~~
   
~~(Note, a couple months ago gradio had a massive update.  For me, this has caused a lot of glitches and errors with extensions; I've briefly tested the Lucid_Vision extension in the newest implementation of textgen and it will work.  However, I was getting timeout popups when vision models were loading for the first time, gradio wasn't waiting for the response from the model upon first load. After a model is loaded once, it is saved in cpu ram cache (this doesn't actively use your ram, it just uses what is free to keep the models in memory so they are quickly reloaded into gpu ram if necessary) and gradio doesn't seem to timeout as often.  The slightly older version of textgen that I've edited does not experience this issue)~~

~~2. Update the transformers library using the cmd_yourOShere.sh/bat file (so either cmd_linux.sh, cmd_macos.sh, cmd_windows.bat, or cmd_wsl.bat) and entering the following lines.  If you run the update wizard after this point, it will overrite this update to transformers.  The newest transformers package has the libraries for paligemma, which the code needs to import regardless of whether or not you are intending to use the model.~~

No longer need to update transformers and the latest version of textgen as of this writing 1.9.1 works well with many of the gradio issues resolved, however gradio might timeout on the page when loading a model for the fisrt time still.  After a model is loaded once, it is saved in cpu ram cache (this doesn't actively use your ram, it just uses what is free to keep the models in memory so they are quickly reloaded into gpu ram if necessary) and gradio doesn't seem to timeout as often.

## Model Information

If you do not want to install DeepseekVL dependencies and are not intending to use deepseek comment out the import lines from the script.py file as displayed here

```
# pip install pexpect
import json
import re
from datetime import datetime  # Import datetime for timestamp generation
from pathlib import Path

import gradio as gr
import torch
from PIL import Image
#from deepseek_vl.models import VLChatProcessor
#from deepseek_vl.utils.io import load_pil_images as load_pil_images_for_deepseek
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, AutoProcessor, \
    PaliGemmaForConditionalGeneration

model_names = ["phiVision", "DeepSeek", "paligemma", "paligemma_cpu", "minicpm_llama3", "bunny"]
```

3. Install **DeepseekVL** if you intend on using that model
   
   Clone the repo: https://github.com/deepseek-ai/DeepSeek-VL into the `repositories` folder of your textgen install

   Open cmd_yourOShere.sh/bat, navigate to the `repositories/DeepSeek-VL` folder via the terminal using `cd your_directory_here` and enter this into the command window:

   ```
   pip install -e .
   ```
   Download the deepseekvl model here: https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat
   
   They have different and smaller models to choose from: https://github.com/deepseek-ai/DeepSeek-VL?tab=readme-ov-file#3-model-downloads

4. If you want to use **Phi-3-vision-128k-instruct**, download it here: https://huggingface.co/microsoft/Phi-3-vision-128k-instruct

5. If you want to use **paligemma-3b**, download it here: https://huggingface.co/google/paligemma-3b-ft-cococap-448 (this is just one out of many fine-tunes google provides)
   
   Read this blog on how to inference with the model: https://huggingface.co/blog/paligemma

6. If you want to use **MiniCPM-Llama3-V-2_5**, download it here: https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5

   The **4-bit** verison of the model can be downloaded here: https://huggingface.co/openbmb/MiniCPM-Llama3-V-2_5-int4

   **Notes about 4-bit MiniCPM:**
   *  It might not look like the model fully unloads from vram, but it does and the vram will be reclaimed if another program needs it
   *  Your directory folder where the model is stored needs to have the term "int4" in it, this is how the extension identifies the 4bit nature of the model

7. If you want to use **Bunny-v1_1-Llama-3-8B-V**, download it here: https://huggingface.co/BAAI/Bunny-v1_1-Llama-3-8B-V
   
    **Notes about Bunny-v1_1-Llama-3-8B-V:**
    *  When running for the first time the model needs internet access so it can download models--google--siglip-so400m-patch14-384 to your cache/huggingface directory.  This is an additional 3.5GB file the model needs to run.

## Updating the config file

8. Before using the extension you need to update the config file; open it in a text editor *Note No quotes around gpu #:
```
{
    "image_history_dir": "(fill_In)/extensions/Lucid_Vision/ImageHistory/",
    "cuda_visible_devices": 0,
    "default_vision_model": "phiVision",
    "phiVision_model_id": "(fill_In)",
    "paligemma_model_id": "(fill_In)",
    "paligemma_cpu_model_id": "(fill_In)",
    "minicpm_llama3_model_id": "(fill_In)",
    "deepseek_vl_model_id": "(fill_in)",
    "bunny_model_id": "(fill_in)"
}
```
If your install directory is /home/username/Desktop/oobLucidVision/text-generation-webui/  the config file will look like this for example:

Make note that you want to change / to \ if you are on Windows

```
   {
    "image_history_dir": "/home/username/Desktop/oobLucidVision/text-generation-webui/extensions/Lucid_Vision/ImageHistory/",
    "cuda_visible_devices": 0,
    "default_vision_model": "phiVision",
    "phiVision_model_id": "(fill_In)", *This is the folder where your phi-3 vision model is stored
    "paligemma_model_id": "(fill_In)", *This is the folder where your paligemma vision model is stored
    "paligemma_cpu_model_id": "(fill_In)", *This is the folder where your paligemma vision model is stored
    "minicpm_llama3_model_id": "(fill_In)", *This is the folder where your minicpm_llama3 vision model is stored, the model can either be the normal fp16 or 4-bit version
    "deepseek_vl_model_id": "(fill_in)",  *This is the folder where your deepseek vision model is stored
    "bunny_model_id": "(fill_in)" *This is the folder where your Bunny-v1_1-Llama-3-8B-V vision model is stored
   }
```

## To work with Alltalk

Alltalk is great!! An extension I use all the time: https://github.com/erew123/alltalk_tts

To get Lucid_Vision to work, the LLM needs to repeat the file directory of an image, and in doing so Alltalk will want to transcribe that text to audo.  It is annoying to have the tts model try an transcribe file directories.  If you want to get Alltalk to work well wtih Lucid_Vision you need to replace the code here in the script.py file that comes with Alltalk (search for the "IMAGE CLEANING" part of the code):

```
########################
#### IMAGE CLEANING ####
########################
OriginalLucidVisionText = ""

# This is the existing pattern for matching both images and file location text
img_pattern = r'<img[^>]*src\s*=\s*["\'][^"\'>]+["\'][^>]*>|File location: [^.]+.png'

def extract_and_remove_images(text):
   """
   Extracts all image and file location data from the text and removes it for clean TTS processing.
   Returns the cleaned text and the extracted image and file location data.
   """
   global OriginalLucidVisionText
   OriginalLucidVisionText = text  # Update the global variable with the original text

   img_matches = re.findall(img_pattern, text)
   img_info = "\n".join(img_matches)  # Store extracted image and file location data
   cleaned_text = re.sub(img_pattern, '', text)  # Remove images and file locations from text
   return cleaned_text, img_info

def reinsert_images(cleaned_string, img_info):
  """
  Reinserts the previously extracted image and file location data back into the text.
  """
  global OriginalLucidVisionText

  # Check if there are images or file locations to reinsert
  if img_info:
     # Check if the "Vision Model Responses:" phrase is present in the original text
     if re.search(r'Vision Model Responses:', OriginalLucidVisionText):
        # If present, return the original text as is, without modifying it
        return OriginalLucidVisionText
     else:
        # If not present, append the img_info to the end of the cleaned string
        cleaned_string += f"\n\n{img_info}"
  return cleaned_string


#################################
#### TTS STANDARD GENERATION ####
#################################
```


## **Quirks and Notes:**
1. **When you load a picture once, it is used once.  Even if the image stays present in the UI element on screen, it is not actively being used.**
2. If you are using other extensions, load Lucid_Vision first.  Put it first in your CMD_FLAGS.txt file or make sure to check it first in the sequence of check boxes in the session tab UI.
3. DeepseekVL can take a while to load initially, that's just the way it is.
   

# **How to use:**

## **BASICS:**

**When you load a picture once, it is used once.  Even if the image stays present in the UI element on screen, it is not actively being used.**

Okay the extension can do many different things with varying levels of difficulty.

Starting out with the basics and understanding how to talk with your vision models:

Scroll down past where you would normally type something to the LLM, you do not need a large language model loaded to use this function.
![image](https://github.com/RandomInternetPreson/Lucid_Vision/assets/6488699/227ff483-5041-46a7-9b5b-a8f9dd3c673e)

Start out by interacting with the vision models without involvement of a seperate LLM model by pressing the `Ask Vision Model` button
![image](https://github.com/RandomInternetPreson/Lucid_Vision/assets/6488699/4530e13f-30a1-43d9-8383-c05e31ddb5d7)

**Note paligemma requries a little diffent type prompting sometimes, read the blog on how to inference with it: https://huggingface.co/blog/paligemma**

Do this with every model you intend on using, upload a picture, and ask a question 


## **ADVANCED:  Updated Tips At End on how to get working with Llama-3-Instruct-8B-SPPO-Iter3 https://huggingface.co/UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3**

Okay, this is why I built the extension in the first place; direct interaction with the vision model was actually an afterthought after I had all the code working.

This is how to use your favorite LLM WITH an additional Vision Model, I wanted a way to give my LLMs eyes essentially.  I realize that good multimodal models are likely around the corner, but until they match the intellect of very good LLMs, I'd rather have a different vision model work with a good LLM.

To use Lucid_Vision as intended requires a little bit of setup:

1. In `Parameters` then `chat` load the "AI_Image" character (this card is in the edited older commit, if using your own version of textgen the character card is here: https://github.com/RandomInternetPreson/Lucid_Vision/blob/main/AI_Image.yaml put it in the `characters` folder of the textgen install folder:
   ![image](https://github.com/RandomInternetPreson/Lucid_Vision/assets/6488699/8b5d770a-b72a-4a74-aa4a-f67fe2a113ba)

2. In `Parameters` then `Generation` under `Custom stopping strings` enter "Vision Model Responses:" exactly as shown:
   ![image](https://github.com/RandomInternetPreson/Lucid_Vision/assets/6488699/7a09650d-4fce-4a8f-badb-b26b4484bf37)

3. Test to see if your LLM understands the instructions for the AI_Image character; ask it this:
   `Can you tell me what is in this image? Do not add any additional text to the beginning of your response to me, start your reply with the questions.`
   ![image](https://github.com/RandomInternetPreson/Lucid_Vision/assets/6488699/817605d7-dcab-4e6c-8c39-6d119a189431)

   I uploaded my image to the Lucid_Vision UI element, then typed my question, then pressed "Generate"; it doesn't matter which order you upload your picture or type your question, both are sent to the LLM at the same time.
   ![image](https://github.com/RandomInternetPreson/Lucid_Vision/assets/6488699/f1525e8d-551d-4719-9e01-3f26b4362d7c)

4. Continue to give the model more images or just talk about something else, in this example I give the model two new images one of a humanoid robot and one of the nutritional facts for a container of sourcream:
   ![image](https://github.com/RandomInternetPreson/Lucid_Vision/assets/6488699/77131490-fb19-4062-a65f-c0880515e252)
   ![image](https://github.com/RandomInternetPreson/Lucid_Vision/assets/6488699/fed7021f-edb5-4667-aae0-c05021291618)
   ![image](https://github.com/RandomInternetPreson/Lucid_Vision/assets/6488699/d1049551-4e48-4ced-85d9-fbb0441330a9)

   Then I asked the model why the sky is blue:

   ![image](https://github.com/RandomInternetPreson/Lucid_Vision/assets/6488699/deb7b70a-edf7-4014-bd8c-2492bae00ea6)

5. Now I can ask it a question from a previous image and the LLM will automatically know to prompt the vision model and it will find the previously saved image on its own:
   ![image](https://github.com/RandomInternetPreson/Lucid_Vision/assets/6488699/e8b0791b-cce8-4183-8971-42e5383d6c46)

   Please note that the vision model received all this as text:
   
   "Certainly! To retrieve the complete contents of the nutritional facts from the image you provided earlier, I will ask the vision model to perform an Optical Character Recognition (OCR) task on the image. Here is the question for the vision model:

    Can you perform OCR on the provided image and extract all the text from the nutrition facts label, including the details of each nutrient and its corresponding values?"

If the vision model is not that smart (Deepseek, paligemma) then it will have a difficult time contextualizing the text prior to the questions for the vision model.

If you run into this situation, it is best to prompt your LLM like this:

`Great, can you get the complete contents of the nutritional facts from earlier, like all the text? Just start with your questions to the vision model please.`
![image](https://github.com/RandomInternetPreson/Lucid_Vision/assets/6488699/38b59679-3550-410d-aac2-29756a058c8f)

6. Please do not get frustrated right away, the Advanced method of usage depends heavily on your LLM's ability to understand the instructions from the character card.  The instructions were written by the 8x22B model itself, I explained what I wanted the model to do and had it write its own instructions.  This might be a viable alternative if you are struggling to get your model to adhere to the instructions from the character card.

You may need to explain things to your llm as your conversation progresses, if it tries to query the vision model when you don't want it to, just explain that to the model and it's unlikely to keep making the same mistake.  

## **Tips on how to get working with Llama-3-Instruct-8B-SPPO-Iter3 https://huggingface.co/UCLA-AGI/Llama-3-Instruct-8B-SPPO-Iter3**

I have found to get this model (and likely similar models) working properly with lucid vision I needed to NOT use the AI_Image character card.  

Instead I started the conversation with the model like this:

![image](https://github.com/user-attachments/assets/4d0ff97c-f9dd-4043-b5be-7b915cb7aae7)

Then copy and pasted the necessary information from the character card:

```
The following is a conversation with an AI Large Language Model. The AI has been trained to answer questions, provide recommendations, and help with decision making. The AI follows user requests. The AI thinks outside the box.

Instructions for Processing Image-Related User Input with Appended File Information:

    Identify the Trigger Phrase: Begin by scanning the user input for the presence of the "File location" trigger phrase. This phrase indicates that the user has selected an image and that the LLM should consider this information in its response.

    Extract the File Path: Once the trigger phrase is identified, parse the text that follows to extract the absolute file path of the image. The file path will be provided immediately after the trigger phrase and will end with the image file extension (e.g., .png, .jpg).

    Understand the Context: Recognize that the user's message preceding the file path is the primary context for the interaction. The LLM should address the user's query or statement while also incorporating the availability of the selected image into its response.

    Formulate Questions for the Vision Model: Based on the user's message and the fact that an image is available for analysis, generate one or more questions that can be answered by the vision model. These questions should be clear, specific, and relevant to the image content.

    Maintain a Conversational Tone: Ensure that the response is natural, coherent, and maintains the flow of the conversation. The LLM should act as an intermediary between the user and the vision model, facilitating a seamless dialogue.

    Prepare the Response Structure: Structure the response so that it includes:

    An acknowledgment of the user's initial message.

    The questions formulated for the vision model, each clearly separated (e.g., by a newline or bullet point).

    Any additional information or clarification requests, if necessary.

    Append the File Path for Processing: At the end of the response, re-append the "File location" trigger phrase along with the extracted file path. This ensures that the subsequent processing steps (such as sending the information to the vision model's CLI) can correctly identify and use the file path.

    Avoid Direct Interaction with the File System: As an LLM, you do not have the capability to directly access or modify files on the server or client systems. Your role is limited to generating text-based responses that include the necessary file path information for other system components to handle.

    Example Response:

   Based on your question about the contents of the image, here are the questions I will ask the vision model:

   - Can you describe the main objects present in the image?
   - Is there any text visible in the image, and if so, what does it say?
   - What appears to be happening in the scene depicted in the image?

   File location: //home/myself/Pictures/rose.png
   
   Do review and contextualize the conversation as it develops (reference your context) to infer if the user is asking new questions of previous images.  Reference the parts of the convesation that are likely to yeild the file location of the image in question, and formulate your response to include that specific file location of that unique .png file, make sure you are referencing the correct .png file as per the part of the conversation that is likely to be in reference to the updated information request.

By following these instructions, the LLM will be able to effectively process user inputs that include image file information and generate appropriate responses that facilitate the interaction between the user and the vision model.
```

Making sure to give a real file location to a real image.

After that it started to pretty much work but I still needed to correct the model (I had to add the spelling error intentionally because if the user sends "File Location:" is messes up the extension:

![image](https://github.com/user-attachments/assets/5090ef8b-88ff-4ec1-a131-ba57e15e9a7d)

The point to take away is that you may need to explain things to your model in various ways for it to contextualize the instructions approprately.
