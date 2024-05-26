WIP DO NOT DOWNLOAD YET

To accurately proportion credit:

WizardLM-2-8x22B (quantized with exllamaV2 to 8bit precision) = 90% of all the work done in this repo.  That model wrote 100% of all the code and the introduction to this repo. 

CommandR+ (quantized with exllamaV2 to 8bit precision) = ~5% of all the work.  CommandR+ contextualized the coding examples and rules for making extensions from Oogabooga's textgen repo extreamly well, and provided a good foundation to develope the code.

RandomInternetPreson = 5% of all the work done.  I came up with the original idea, the original general outline of how the pieces would interact, and provided feedback to the WizardLM model, but I did not write any code.  I'm actually not very good with python yet, with most of my decades of coding being in Matlab.

My goal from the beginning was to write this extension offline without any addiitonal resources, sometimes it was a little frusturating but I soon understood how to get want I needed from the models running locally.

I would say that most of the credit should go to Oobabooga, for without them I would be struggling to even interact with my models.  Please consider supporting them:

https://github.com/sponsors/oobabooga

or 

https://ko-fi.com/oobabooga

I am their top doner on ko-fi (Mr. A) and donate 15$ montly, their software is extreamly important to the opensource community.


# Lucid_Vision Extension for Oobabooga's textgen-webui

Welcome to the Lucid Vision Extension repository! This extension enhances the capabilities of textgen-webui by integrating advanced vision models, allowing users to have contextualized conversations about images with their favorite language models; and allowing direct communciation with vision models.

## Features

* Multi-Model Support: Interact with different vision models, including PhiVision, DeepSeek, and PaliGemma, with options for both GPU and CPU inference.

* On-Demand Loading: Vision models are loaded into memory only when needed to answer a question, optimizing resource usage.

* Seamless Integration: Easily switch between vision models using a Gradio UI radio button selector.

* Cross-Platform Compatibility: The extension is designed to work on various operating systems, including Unix-based systems and Windows. (not tested in Windows yet, but should probably work?)

* Direct communication with vision models, you do not need to load a LLM to interact with the seperate vision models.

## How It Works

The Lucid Vision Extension operates by intercepting and modifying user input and output within the textgen-webui framework. When a user uploads an image and asks a question, the extension appends a special trigger phrase ("File location") and extracts the associated file path and question.

So if a user enters text into the "send a message" field and has a new picture uploaded into the Lucid_Vision ui, what will happen behind the scenes is the at the user message will be appended with the "File Location: (file location)" Trigger phrase, at which point the LLM will see this and understand that it needs to reply back with questions about the image, and that those questions are being sent to a vison model.

The cool thing is that let's say later in the conversation you want to know something specific about a previous picture, all you need to do is ask your LLM, YOU DO NOT NEED TO REUPLOAD THE PICTURE, the LLM should be able to interact with the extension on its own after you uploaed your first picture.

Depending on the selected vision model, the extension either sends a command to the model's command-line interface (for DeepSeek) or directly interacts with the model's Python API (for PhiVision and PaliGemma) to generate a response. The response is then appended to the chat history, providing the user with detailed insights about the image.

The extension is designed to be efficient with system resources by only loading the vision models into memory when they are actively being used to process a question. After generating a response, the models are immediately unloaded to free up memory and GPU VRAM.


## **How to install and setup:**

1. Install this edited prior commit from oobabooga's textgen https://github.com/RandomInternetPreson/textgen_webui_Lucid_Vision_Testing OR use the latest version of textgen.  If using the edited older version, make sure to rename the install folder `text-generation-webui`
   
(Note, a couple months ago gradio had a massive update.  For me, this has caused a lot of glitches and errors with extensions; I've briefly tested the Lucid_Vision extension in the newest implementaion of textgen and it will work.  However, I was getting timeout popups when vision models were loading for the first time, gradio wasn't watiing for the response from the model upon first load. After a model is loaded once, it is saved in cpu ram cache (this doesn't actively use your ram, it just uses what is free to keep the models in memory so they are quickly reloaded into gpu ram if necessary) and gradio doesn't seem to timeout as often.  The slightly older version of textgen that I've edited does not experience this issue)

2. If you want to use the update wizard, right now would be the time to install the requirements for Lucid_Vision with the update wizard; so install lucid vision as you normally would any other extension.
   
2a. If you want just want to install the one extra dependency Lucid_Vision requires, then using the cmd_yourOShere.sh/bat file (so either cmd_linux.sh, cmd_macos.sh, cmd_windows.bat, or cmd_wsl.bat) and entering the following line.

```
pip install pexpect
```

3. Update the transformers library using the cmd_yourOShere.sh/bat file (so either cmd_linux.sh, cmd_macos.sh, cmd_windows.bat, or cmd_wsl.bat) and entering the following lines.  If you run the update wizard after this point, it will overrite this update to transformers.  The newest transformes package has the libraries for paligemma, which the code needs to import regardless of whether or not you are intending to use the model.

```
pip uninstall transformers -y

pip install transformers --upgrade --no-cache-dir
```

4. Install DeepseekVL if you intend on using that model
   
   Clone the repo: https://github.com/deepseek-ai/DeepSeek-VL into the `repositories` folder of your textgen install

   Open cmd_yourOShere.sh/bat, navigate to the `repositories/DeepSeek-VL` folder via the terminal using `cd your_directory_here` and enter this into the command window:

   ```
   pip install -e .
   ```
   Download the deepseekvl model here: https://huggingface.co/deepseek-ai/deepseek-vl-7b-chat
   
   They have different and smaller models to choose from: https://github.com/deepseek-ai/DeepSeek-VL?tab=readme-ov-file#3-model-downloads

6. If you want to use Phi-3-vision-128k-instruct, download it here: https://huggingface.co/microsoft/Phi-3-vision-128k-instruct

7. If you want to use paligemma-3b, download it here: https://huggingface.co/google/paligemma-3b-ft-cococap-448 (this is just one out of many fine-tunes google provides)
   
   Read this blog on how to inference with the model: https://huggingface.co/blog/paligemma

9. Before using the extension you need to update the config file; open it in a text editor:
```
   {
    "image_history_dir": "(fill_In)/extensions/Lucid_Vision/ImageHistory/",
    "python_exec": "(fill_In)/installer_files/env/bin/python",
    "cli_script_path": "(fill_In)/repositories/DeepSeek-VL/cli_chat.py",
    "model_path": "(fill_In)",
    "cuda_visible_devices": "0",
    "default_vision_model": "phiVision",
    "phiVision_model_id": "(fill_In)",
    "paligemma_model_id": "(fill_In)",
    "paligemma_cpu_model_id": "(fill_In)"
   }
```
If your install directory is /home/username/Desktop/oobLucidVision/text-generation-webui/  the config file will look like this for example:

Make note that you want to change / to \ if you are on Windows

```
   {
    "image_history_dir": "/home/username/Desktop/oobLucidVision/text-generation-webui/extensions/Lucid_Vision/ImageHistory/",
    "python_exec": "/home/username/Desktop/oobLucidVision/text-generation-webui/installer_files/env/bin/python",
    "cli_script_path": "/home/username/Desktop/oobLucidVision/text-generation-webui/repositories/DeepSeek-VL/cli_chat.py",
    "model_path": "(fill_In)",  *This is the folder where your deepseekvl model is stored
    "cuda_visible_devices": "0",
    "default_vision_model": "phiVision",
    "phiVision_model_id": "(fill_In)", *This is the folder where your phi-3 vision model is stored
    "paligemma_model_id": "(fill_In)", *This is the folder where your paligemma vision model is stored
    "paligemma_cpu_model_id": "(fill_In)" *This is the folder where your paligemma vision model is stored
   }
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


## **ADVANCED:**

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


