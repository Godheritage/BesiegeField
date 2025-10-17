import json
from pathlib import Path
import re
import sys

PROMPTBASE_PATH = Path(__file__).resolve().parent/ "prompts"

def read_file(file_path, file_type="text"):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            if file_type == "json":
                
                json_str =json.load(file)
                json_str = json.dumps(json_str, ensure_ascii=False, indent=None, separators=(',', ':')).replace('\n', '').replace('{', '{{').replace('}', '}}')
                return str(json_str)
            else:
                
                content = file.read()
                return content.strip()  
    except FileNotFoundError:
        print(f"file not found:{file_path}")
    except json.JSONDecodeError as e:
        print(f"JSON format error:{e}")
    except Exception as e:
        print(f"Reading file error:{e}")
    return None

def get_prompt(prompt_content, kwargs):
    return prompt_content.format(**kwargs)

class LocalizationDict(dict):
    def __getitem__(self, key):
        return self.get(key, key) 
    def update_key(self, key, new_value):
        self[key] = new_value


metadesigner_prompt = {
    "deafult_en": {
        "prompt": """
        You are a mechanical designer, and your task is to design a machine in the game Besiege based on the user's requirements. Please gain a general understanding of the game based on the following information:
        I. Game Introduction:
        {game_intro}
        II. Block Introduction:
        {block_intro}
        III. Mechanical Design Requirements:
        {design_requirments}
        IV. Output Format Requirements:
        {output_format}
        V. Note:
        {warning}

        I will provide the user input below, 
        please generate a mechanical overview in JSON format based on the user's description.
        """,
        "example":
        [],
        "kargs":{
            "game_intro": PROMPTBASE_PATH / "metadesigner_knowledge/deafult_en/game_intro.txt",
            "block_intro": PROMPTBASE_PATH / "metadesigner_knowledge/deafult_en/block_intro.txt",
            "output_format": PROMPTBASE_PATH / "metadesigner_knowledge/deafult_en/output_format.txt",
            "design_requirments": PROMPTBASE_PATH / "metadesigner_knowledge/deafult_en/design_requirments.txt",
            "warning": PROMPTBASE_PATH / "metadesigner_knowledge/deafult_en/warning.txt"
        },  
    },
    "ablation_en": {
        "prompt": """
        You are a mechanical designer, and your task is to design a machine in the game Besiege based on the user's requirements. Please gain a general understanding of the game based on the following information:
        I. Game Introduction:
        {game_intro}
        II. Block Introduction:
        {block_intro}
        III. Mechanical Design Requirements:
        {design_requirments}
        IV. Output Format Requirements:
        {output_format}
        V. Note:
        {warning}

        I will provide the user input below, 
        please generate a mechanical overview in JSON format based on the user's description.
        """,
        "example":
        [],
        "kargs":{
            "game_intro": PROMPTBASE_PATH / "metadesigner_knowledge/ablation_en/game_intro.txt",
            "block_intro": PROMPTBASE_PATH / "metadesigner_knowledge/ablation_en/block_intro.txt",
            "output_format": PROMPTBASE_PATH / "metadesigner_knowledge/ablation_en/output_format.txt",
            "design_requirments": PROMPTBASE_PATH / "metadesigner_knowledge/ablation_en/design_requirments.txt",
            "warning": PROMPTBASE_PATH / "metadesigner_knowledge/ablation_en/warning.txt"
        }
    }
}

designer_prompt = {
    "short_abl_3drep_en": {
        "prompt": """
        You are a mechanical builder in the game "Besiege." 
        Your task is to add new blocks to an existing machine structure according to user requests and finally output the complete machine JSON data.
        I. Game Introduction:
        {game_intro}
        II. Introduction to Blocks:
        {block_intro}
        III. Introduction to JSON Format:
        {json_intro}
        IV. Construction Guidance:
        {build_guidance}
        V. Output Format Requirements:
        {output_format}
        VI. Note:
        {warning}
        Next I will provide user input, please generate a JSON based on the description.
        """,
        "example":
        [],
        "kargs":{
            "game_intro": PROMPTBASE_PATH / "designer_knowledge/short_abl_3drep_en/game_intro.txt",
            "block_intro": PROMPTBASE_PATH / "designer_knowledge/short_abl_3drep_en/block_intro.txt",
            "json_intro": PROMPTBASE_PATH / "designer_knowledge/short_abl_3drep_en/json_intro.txt",
            "build_guidance": PROMPTBASE_PATH / "designer_knowledge/short_abl_3drep_en/build_guidance.txt",
            "warning": PROMPTBASE_PATH / "designer_knowledge/short_abl_3drep_en/warning.txt",
            "output_format": PROMPTBASE_PATH / "designer_knowledge/short_abl_3drep_en/output_format.txt"
        }   
    },
    "short_en": {
        "prompt": """
        You are a mechanical builder in the game "Besiege." 
        Your task is to add new blocks to an existing machine structure according to user requests and finally output the complete machine JSON data.
        I. Game Introduction:
        {game_intro}
        II. Introduction to Blocks:
        {block_intro}
        III. Introduction to JSON Format:
        {json_intro}
        IV. Construction Guidance:
        {build_guidance}
        V. Output Format Requirements:
        {output_format}
        VI. Note:
        {warning}
        Next I will provide user input, please generate a JSON based on the description.
        """,
        "example":
        [],
        "kargs":{
            "game_intro": PROMPTBASE_PATH / "designer_knowledge/short_en/game_intro.txt",
            "block_intro": PROMPTBASE_PATH / "designer_knowledge/short_en/block_intro.txt",
            "json_intro": PROMPTBASE_PATH / "designer_knowledge/short_en/json_intro.txt",
            "build_guidance": PROMPTBASE_PATH / "designer_knowledge/short_en/build_guidance.txt",
            "warning": PROMPTBASE_PATH / "designer_knowledge/short_en/warning.txt",
            "output_format": PROMPTBASE_PATH / "designer_knowledge/short_en/output_format.txt"
        }   
    },
    "ablation_singleagent_en":{
        "prompt": """
        {single_agent}
        """,
        "example":
        [],
        "kargs":{
            "single_agent": PROMPTBASE_PATH / "designer_knowledge/ablation_singleagent_en/single_agent.txt",
        }
    }
}

inspector_prompt = {
    "short_en": {
        "prompt": """
        I'll provide you with a mission in the game Besiege, 
        along with the machine designed for it in JSON format and its 3D information. 
        Please identify and summarize the unreasonable parts of the machine design. 
        Here's the introduction to the game and construction knowledge.
        I. Game Introduction:
        {game_intro}
        II. Introduction to Blocks:
        {block_intro}
        III. Introduction to JSON and 3D Information:
        {input_intro}
        IV. Introduction to Output Format:
        {output_intro}
        V. Notes:
        {warning}
        Below, I will provide you with JSON and 3D information. Please answer the user's question based on this information.
        """,
        "example":
        [],
        "kargs":{
            "game_intro": PROMPTBASE_PATH / "inspector_knowledge/short_en/game_intro.txt",
            "block_intro": PROMPTBASE_PATH / "inspector_knowledge/short_en/block_intro.txt",
            "input_intro": PROMPTBASE_PATH / "inspector_knowledge/short_en/input_intro.txt",
            "output_intro":PROMPTBASE_PATH / "inspector_knowledge/short_en/output_intro.txt",
            "warning": PROMPTBASE_PATH / "inspector_knowledge/short_en/warning.txt"
        },
        "userquestion":"""
            Task Introduction
            {task_definition}
            JSON Information
            {json_file}
            3D Information
            {threedinfo}
            Mechanical Structure Information
            {structure_info}
            Initial State of the Machine
            The machine is initially placed on the ground, facing in the z+ direction, with the target direction being z+.
            
            Questions:
            1. Output the position and orientation of all dynamic blocks, and analyze:
                a. The impact of dynamic blocks on the machine
                b. The direction of force provided by dynamic blocks
                c. The impact on sub-blocks and the direction of force on the machine

            2. Output static blocks other than basic structural blocks, and analyze the rationality of their orientation and position.

            3. Balance Check (self-gravity)
                a. The center of gravity of the machine (find the block closest to the center of gravity)
                b. Whether parts of the machine will sink due to gravity
            4. Comprehensive Analysis
                a. Summarize the direction of all forces to analyze the movement of the machine
                b. Identify logically unreasonable blocks, output their hierarchical structure and reasons for unreasonableness
            """,
        
    },
    "short-auto-3dinfo_en": {
        "prompt": """
        I'll provide you with a mission in the game Besiege, 
        along with the machine designed for it in JSON format and its 3D information. 
        Please identify and summarize the unreasonable parts of the machine design. 
        Your report will help next agent modify the machine.
        Here's the introduction to the game and construction knowledge.
        I. Game Introduction:
        {game_intro}
        II. Introduction to Blocks:
        {block_intro}
        III. Introduction to JSON and 3D Information:
        {input_intro}
        IV. Introduction to Output Format:
        {output_intro}
        V. Notes:
        {warning}
        Below, I will provide you with JSON and 3D information. Please answer the user's question based on this information.
        """,
        "example":
        [],
        "kargs":{
            "game_intro": PROMPTBASE_PATH / "inspector_knowledge/short-auto-3dinfo_en/game_intro.txt",
            "block_intro": PROMPTBASE_PATH / "inspector_knowledge/short-auto-3dinfo_en/block_intro.txt",
            "input_intro": PROMPTBASE_PATH / "inspector_knowledge/short-auto-3dinfo_en/input_intro.txt",
            "output_intro":PROMPTBASE_PATH / "inspector_knowledge/short-auto-3dinfo_en/output_intro.txt",
            "warning": PROMPTBASE_PATH / "inspector_knowledge/short-auto-3dinfo_en/warning.txt"
        },
        "userquestion":"""
            Task Introduction
            {task_definition}
            JSON Information
            {json_file}
            3D Information
            {threedinfo}
            Mechanical Structure Information
            {structure_info}
            Initial State of the Machine
            The machine is initially placed on the ground, facing in the z+ direction, with the target direction being z+.
            
            Questions:
            1. Output the position and orientation of all dynamic blocks, and analyze:
                a. The impact of dynamic blocks on the machine
                b. The direction of force provided by dynamic blocks
                c. The impact on sub-blocks and the direction of force on the machine

            2. Output static blocks other than basic structural blocks, and analyze the rationality of their orientation and position.

            3. Balance Check (self-gravity)
                a. The center of gravity of the machine (find the block closest to the center of gravity)
                b. Whether parts of the machine will sink due to gravity
            4. Comprehensive Analysis
                a. Summarize the direction of all forces to analyze the movement of the machine
                b. Identify logically unreasonable blocks, output their hierarchical structure and reasons for unreasonableness
            """,
        
    }
    
}

refiner_prompt = {
    "short_en": {
        "prompt": """
        I will give you a task in the game Besiege, as well as the 3D information of the machine designed to complete this task. There are some unreasonable aspects in the design of this machine, and I would like you to modify these parts:
        I. Game Introduction:
        {game_intro}
        II. Block Introduction:
        {block_intro}
        III. Input Introduction:
        {input_intro}
        IV. Modification Method Introduction:
        {modify_guide}
        V. Output Format Introduction:
        {output_intro}
        VI. Note:
        {warning}
        Below, I will provide you with the json and 3D information. Please modify the machine accordingly.
        """,
        "example":
        [],
        "kargs":{
            "game_intro": PROMPTBASE_PATH / "refiner_knowledge/short_en/game_intro.txt",
            "block_intro": PROMPTBASE_PATH / "refiner_knowledge/short_en/block_intro.txt",
            "input_intro": PROMPTBASE_PATH / "refiner_knowledge/short_en/input_intro.txt",
            "output_intro":PROMPTBASE_PATH / "refiner_knowledge/short_en/output_intro.txt",
            "modify_guide":PROMPTBASE_PATH / "refiner_knowledge/short_en/modify_guide.txt",
            "warning": PROMPTBASE_PATH / "refiner_knowledge/short_en/warning.txt"
        },
        "userquestion":"""
            Task Introduction
            {task_definition}
            Machine JSON Structure
            {jsoninfo}
            3D Information
            {threedinfo}
            Machine Defective Structure
            {focus_info}
            Machine Defective Structure Report
            {focus_report}
            Machine Initial State
            The machine is initially placed on the ground, facing the z+ direction, with the target direction being z+.
            Please provide the modification steps to help improve the machine.
            """, 
    },
    "env_short_en": {
        "prompt": """
        I will give you a task in the game Besiege, as well as the information of the machine designed to complete this task. 
        The machine failed the task. 
        Please modify the machine according to the fail feedback.
        I. Game Introduction:
        {game_intro}
        II. Block Introduction:
        {block_intro}
        III. Input Introduction:
        {input_intro}
        IV. Modification Method Introduction:
        {modify_guide}
        V. Output Format Introduction:
        {output_intro}
        VI. Note:
        {warning}
        Below, I will provide you with the json, 3D information and the task fail feedback. Please modify the machine accordingly.
        """,
        "example":
        [],
        "kargs":{
            "game_intro": PROMPTBASE_PATH / "refiner_knowledge/env_short_en/game_intro.txt",
            "block_intro": PROMPTBASE_PATH / "refiner_knowledge/env_short_en/block_intro.txt",
            "input_intro": PROMPTBASE_PATH / "refiner_knowledge/env_short_en/input_intro.txt",
            "output_intro":PROMPTBASE_PATH / "refiner_knowledge/env_short_en/output_intro.txt",
            "modify_guide":PROMPTBASE_PATH / "refiner_knowledge/env_short_en/modify_guide.txt",
            "warning": PROMPTBASE_PATH / "refiner_knowledge/env_short_en/warning.txt"
        },
        "userquestion":"""
            Task Introduction
            {task_definition}
            Mechanical JSON Structure
            {jsoninfo}
            3D Information
            {threedinfo}
            Environment Feedback  
            {env_feedback}
            Initial State of the Machine
            The machine is initially placed on the ground, facing the z+ direction, with the target direction being z+.
            Historical Modification Information
            {fail_history}
            Please remember that the task failure is due to the mechanical design defects. 
            Your ultimate goal is to modify the mechanical design defects so that it can complete the task.
            The historical modification information represents your previous attempts to make changes, and these steps did not comply with the game rules.
            Based on the information about the task failure, please make modifications to the mechanics. 
            When designing the modification steps, avoid making the same mistakes as those in the historical modification information.
            """, 
    }
}

env_querier_prompt = {
    "short_en": {
        "prompt": """
        I will give you a task in the game Besiege, as well as the information of the machine designed to complete this task. 
        The machine has finished the task simulation and returned some environmental feedback to describe its performance.
        Please analyze the issues with the machine based on the feedback and request more feedback if needed.
        I. Game Introduction:
        {game_intro}
        II. Block Introduction:
        {block_intro}
        III. Input Introduction:
        {input_intro}
        IV. Query Introduction:
        {modify_guide}
        V. Output Format Introduction:
        {output_intro}
        VI. Note:
        {warning}
        Below, I will provide you with the json and 3D information, as well as the environmental feedback. Please request more feedback as needed.
        """,
        "example":
        [],
        "kargs":{
            "game_intro": PROMPTBASE_PATH / "env_querier_knowledge/short_en/game_intro.txt",
            "block_intro": PROMPTBASE_PATH / "env_querier_knowledge/short_en/block_intro.txt",
            "input_intro": PROMPTBASE_PATH / "env_querier_knowledge/short_en/input_intro.txt",
            "output_intro":PROMPTBASE_PATH / "env_querier_knowledge/short_en/output_intro.txt",
            "modify_guide":PROMPTBASE_PATH / "env_querier_knowledge/short_en/modify_guide.txt",
            "warning": PROMPTBASE_PATH / "env_querier_knowledge/short_en/warning.txt"
        },
        "userquestion":"""
            Task Introduction
            {task_definition}
            Mechanical JSON Structure
            {jsoninfo}
            3D Information
            {threedinfo}
            Environment Feedback  
            {env_feedback}
            Initial State of the Machine
            The machine is initially placed on the ground, facing the z+ direction, with the target direction being z+.
            """, 
    }
}

single_prompt = {
    "short_en": {
        "prompt": """
        You are a mechanical builder in the game "Besiege." 
        Your task is to output the complete machine JSON data according to user requests.
        I. Game Introduction:
        {game_intro}
        II. Introduction to Blocks:
        {block_intro}
        III. Introduction to JSON Format:
        {json_intro}
        IV. Construction Guidance:
        {build_guidance}
        V. Output Format Requirements:
        {output_format}
        VI. Note:
        {warning}
        Next I will provide user input, please generate a JSON based on the description.
        """,
        "example":
        [],
        "kargs":{
            "game_intro": PROMPTBASE_PATH / "single_knowledge/short_en/game_intro.txt",
            "block_intro": PROMPTBASE_PATH / "single_knowledge/short_en/block_intro.txt",
            "json_intro": PROMPTBASE_PATH / "single_knowledge/short_en/json_intro.txt",
            "build_guidance": PROMPTBASE_PATH / "single_knowledge/short_en/build_guidance.txt",
            "warning": PROMPTBASE_PATH / "single_knowledge/short_en/warning.txt",
            "output_format": PROMPTBASE_PATH / "single_knowledge/short_en/output_format.txt"
        }   
    },
}

def format_agentic_systemprompt(agent,prompt_type="short",block_limitations=None,need_example = False):
    agentic_map={
        "meta-designer":metadesigner_prompt,
        "designer":designer_prompt,
        "inspector":inspector_prompt,
        "refiner":refiner_prompt,
        "env_querier":env_querier_prompt,
        "single":single_prompt
    }
    prompt_type= prompt_type+"_en"
    
    prompt_infos = agentic_map[agent][prompt_type]
    system_prompt = prompt_infos["prompt"]
    
    kwargs = {}
    
    for key in prompt_infos["kargs"].keys():
        file_type = Path(prompt_infos["kargs"][key]).suffix.replace(".", "")
        if key == "block_intro" and agent!="meta-designer":
            block_infos = read_file(prompt_infos["kargs"][key], file_type=file_type)
            match = re.match(r'^(.*?)```json\n(.*?)```', block_infos, re.DOTALL)
            main_content = match.group(1).strip()
            json_data = json.loads(match.group(2).strip())
            limited_block_infos = [info for info in json_data if info["Type ID"] in block_limitations]
            kwargs[key] = f"{main_content}\n{limited_block_infos}\n"
        else:
            kwargs[key] = read_file(prompt_infos["kargs"][key], file_type=file_type)
        
        
    formatted_prompt = get_prompt(
        system_prompt,
        kwargs=kwargs
    )
    if need_example:
        example = prompt_infos["example"]
    else:
        example=[]

    return formatted_prompt,example     

def format_agentic_input(kargs, agent_type, prompt_type="deafult"):
    prompt_map = {
        "refiner": refiner_prompt,
        "env_querier": env_querier_prompt,
        "inspector": inspector_prompt
    }
    if agent_type not in prompt_map:
        raise ValueError(f"Unsupported agent_type: {agent_type}")
    prompt_dict = prompt_map[agent_type]

    prompt_type = prompt_type + "_en"


    formatted_prompt = get_prompt(
        prompt_dict[prompt_type]["userquestion"],
        kwargs=kargs
    )
    return formatted_prompt


if __name__ == "__main__":
    pass