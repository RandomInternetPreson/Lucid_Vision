WIP Do NOT download yet


Basics

1. Install this edited prior commit from oobabooga's textgen https://github.com/RandomInternetPreson/textgen_webui_Lucid_Vision_Testing OR use the latest version of textgen.  Is using the edited older version, make sure to rename the install folder `text-generation-webui`
   
(Note, a couple months ago gradio had a massive update.  For me, this has caused a lot of glitches and errors with extensions; I've briefly tested the Lucid_Vision extension in the newest implementaion of textgen and it will work.  However, I was getting timeout popups when vision models were loading for the first time, gradio wasn't watiing for the response from the model upon first load. After a model is loaded once, it is saved in cpu ram cache (this doesn't actively use your ram, it just uses what is free to keep the models in memory so they are quicly reloaded into gpu ram if necessary) and gradio doesn't seem to timeout as often.  The slightly older version of textgen that I've edited does not experience this issue)

2. If you want to use the update wizard, right now would be the time to install the requirements for Lucid_Vision with the update wizard; so install lucid vision as you normally would any other extension.
   
2a. If you want just want to install the one extra dependency Lucid_Vision requires, then using the cmd_yourOShere.sh/bat file (so either cmd_linux.sh, cmd_macos.sh, cmd_windows.bat, or cmd_wsl.bat) and entering the following line.

```
pip install pexpect
```

3. Update the transformers library using the cmd_yourOShere.sh/bat file (so either cmd_linux.sh, cmd_macos.sh, cmd_windows.bat, or cmd_wsl.bat) and entering the following lines.  If you run the update wizard after this point, it will overrite this update to transformers.  The newest transformes package has the libraries for paligemma, which the code needs to import regardless of wheather or not you are intending to use the model.

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

8. Before using the extension you need to update the config file; open it in a text editor:
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
If your install directory is /home/username/Desktop/oobLucidVision/text-generation-webui/  Make note that you want to change / to \ if you are on Windows

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
