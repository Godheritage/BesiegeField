# encoding=utf-8
import os
import re
import base64
import requests
from langchain_openai import AzureChatOpenAI
from openai import OpenAI,AzureOpenAI
from langchain.prompts.chat import ChatPromptTemplate
from typing import List, Dict, Union
from openai._types import NOT_GIVEN

def get_float_from_text(text):
    matches = re.findall(r"\d+\.\d+", text) + re.findall(r"\d+", text)
    if len(matches) == 0:
        return 0
    return float(matches[0])


class TextRefiner:
    def __init__(
        self,
        definition=None,
        model_name="gpt-3.5-turbo",
        temperature=0.7,
        samples=[],
        historical_num=6,
        api_version=None,  
        CoT=False,  
        max_tokens = 2000,
        top_p=0.95
    ):
        self.definition = definition or ""
        self.model_name = model_name
        if api_version==None:
            self.llm = OpenAI(api_key=os.environ['OPENAI_API_KEY'],base_url=os.environ['OPENAI_API_BASE'])
        else:
            self.llm = AzureOpenAI(
                api_key=os.environ['AZURE_OPENAI_API_KEY'],
                azure_endpoint=os.environ['AZURE_OPENAI_ENDPOINT'],
                api_version=api_version,  
            )
        self.samples = samples
        self.historical_messages = []
        self.historical_num = historical_num
        self.CoT = CoT 
        self.temperature = temperature
        self.top_p = top_p
        if "o3" in model_name:
            self.top_p=None
        self.max_tokens = max_tokens

    def encode_image(self, image_path: str) -> str:
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def get_image_url_content(self, image_url: str) -> Dict:
        response = requests.get(image_url)
        if response.status_code == 200:
            return {
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64.b64encode(response.content).decode('utf-8')}"
                },
            }
        else:
            raise ValueError(f"Can not get image: {image_url}")

    def refine_with_image(
        self, input_text: str, image_path: str = None, image_url: str = None
    ) -> str:
        messages = [{"role": "system", "content": self.definition}]
        # messages.extend(self.samples)
        # messages.extend(self.historical_messages)
        human_template = [{"type": "text", "text": input_text}]
        if image_path:
            base64_image = self.encode_image(image_path)
            human_template.append(
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    },
                }
            )
        elif image_url:
            human_template.append(self.get_image_url_content(image_url))

        messages.append({"role": "user", "content": human_template})
        

        response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p
        )
        content = response.choices[0].message.content
        self.historical_messages.append({"role": "user", "content": human_template})
        self.historical_messages.append({"role": "assistant", "content": content})


        if (len(self.historical_messages)) > self.historical_num:
            self.historical_messages = self.historical_messages[-self.historical_num :]
            if self.historical_messages[0]["role"]=="assistant":
                self.historical_messages.pop(0)
        
        if self.CoT:
            reasoning_content = response.choices[0].message.reasoning_content
            return {"content": content, "reasoning_content": reasoning_content}
        else:
            return {"content": content}

    def filter_each_line(self, input_text):
        output_lines = []
        for line in input_text.split("\n"):
            output_line = self.refine(line)
            prob = get_float_from_text(output_line)
            print(line, " - ", output_line, " - ", prob)
            if prob <= 0.3:
                continue
            output_lines.append(line)
        return "\n".join(output_lines)

    def filter_each_line_file(self, input_path, output_path):
        with open(input_path, encoding="utf-8", mode="r") as f:
            input_text = f.read()
        output_text = self.filter_each_line(input_text)
        with open(output_path, encoding="utf-8", mode="w") as f:
            f.write(output_text)

    def refine_each_line(self, input_text):
        output_lines = []
        for line in input_text.split("\n"):
            output_line = self.refine(line)
            if "null" in output_line:
                print(line, " - ", output_line)
                continue
            print(line, " - ", output_line)
            output_lines.append(output_line)
        return "\n".join(output_lines)

    def refine_each_line_file(self, input_path, output_path):
        with open(input_path, encoding="utf-8", mode="r") as f:
            input_text = f.read()
        output_text = self.refine_each_line(input_text)
        with open(output_path, encoding="utf-8", mode="w") as f:
            f.write(output_text)

    def refine(self, input_text):

        messages = [{"role": "system", "content": self.definition}]
        # messages.extend(self.samples)
        # messages.extend(self.historical_messages)

        if "gemini" in self.model_name:
            extra_body={
            'extra_body': {
                "google": {
                "thinking_config": {
                    "thinking_budget": 8000,
                    "include_thoughts": True
                }
                }
            }
            }
        else: extra_body=None
        
        if "gpt-5" in self.model_name:
            reasoning_effort="high"
            self.temperature =1.0
            self.top_p=NOT_GIVEN
        else:
            reasoning_effort=NOT_GIVEN
        
        messages.append({"role": "user", "content": input_text})
        response = self.llm.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
            top_p=self.top_p,
            extra_body=extra_body,
            reasoning_effort=reasoning_effort,
        )

        content = response.choices[0].message.content
        self.historical_messages.append({"role": "user", "content": input_text})
        self.historical_messages.append({"role": "assistant", "content": content})

        if (len(self.historical_messages)) > self.historical_num:
            self.historical_messages = self.historical_messages[-self.historical_num :]
            if self.historical_messages[0]["role"]=="assistant":
                self.historical_messages.pop(0)
        
        if self.CoT:
            try:
                reasoning_content = response.choices[0].message.reasoning_content
                return {"content": content, "reasoning_content": reasoning_content,"history":messages}
            except:
                return False
        else:
            return {"content": content,"history":messages}


    def refine_file(self, input_path, output_path):
        with open(input_path, encoding="utf-8", mode="r") as f:
            input_text = f.read()
        output_text = self.refine(input_text)
        with open(output_path, encoding="utf-8", mode="w") as f:
            f.write(output_text)

    def set_definition(self, definition):
        self.definition = definition