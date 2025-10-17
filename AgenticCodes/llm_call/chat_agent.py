# encoding=utf-8
import os
import xml.etree.ElementTree as ET
import json
import sys

import threading
_env_lock = threading.Lock()

def load_chat_agent(chat_agent_path, model_name="deepseek-reasoner"):
    with _env_lock:
        for var in ["AZURE_OPENAI_API_KEY", "AZURE_OPENAI_ENDPOINT", 
                   "OPENAI_API_KEY", "OPENAI_API_BASE"]:
            if var in os.environ:
                del os.environ[var]

        with open(chat_agent_path, "r") as f:
            content = json.load(f)
        
        chat_agent_info = content[model_name]
        Azure = chat_agent_info.get("Azure", False)
        api_version = chat_agent_info.get("api_version", None)
        CoT = chat_agent_info.get("CoT", False)
        
        if isinstance(CoT, str):
            CoT = CoT.lower() == "true"
        if isinstance(Azure, str):
            Azure = Azure.lower() == "true"

        if Azure:
            os.environ['AZURE_OPENAI_API_KEY'] = chat_agent_info["AZURE_OPENAI_API_KEY"]
            os.environ['AZURE_OPENAI_ENDPOINT'] = chat_agent_info["AZURE_OPENAI_ENDPOINT"]
        else:
            os.environ['OPENAI_API_KEY'] = chat_agent_info["OPENAI_API_KEY"]
            os.environ['OPENAI_API_BASE'] = chat_agent_info["OPENAI_API_BASE"]

    return model_name, api_version, CoT




def agent_translator_once(agent_translator,glosses,image=None):
    if len(glosses) == 0:
        return ""
    if image==None:
      return agent_translator.refine(glosses)
    return agent_translator.refine_with_image(glosses,image_path=image)

def agent(agent_translator,glosses,image=None):
    return agent_translator_once(agent_translator,glosses,image)

if __name__ == '__main__':
    pass

  
    
    
    
    
    
    