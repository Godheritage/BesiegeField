# utils.py
import secrets
import json
import os
import re
import ast
from copy import deepcopy
import numpy as np
from scipy.spatial.transform import Rotation as R
from AgenticCodes.config import BLOCKPROPERTY

def parse_data(data):
    KEYWORD_MAPPING = {
        "throwing distance": ("throwing_distance", float),
        "car driving distance": ("throwing_distance", float),
        "max speed": ("max_speed", float),
        "max height": ("max_height", float),
        "average speed": ("average_speed", eval),
        "actual position": ("actual_position", eval),
        "startingBlock": ("startingBlock_position", eval),
        "land_idx": ("land_idx", int)
    }
    
    result = {key: None for key in set(k[0] for k in KEYWORD_MAPPING.values())}
    
    lines = [line.strip() for line in data.strip().split('\n') if line.strip()]
    
    for i, line in enumerate(lines):
        for keyword, (result_key, processor) in KEYWORD_MAPPING.items():
            if keyword in line and i + 1 < len(lines):
                value = lines[i + 1].strip()
                try:
                    result[result_key] = processor(value)
                except (ValueError, SyntaxError):
                    continue
                break  
    
    return result


def blouder_throw_scoring(env_feedback):
    not_dilvery=1
    enough_height=1
    height_ratio=1.0
    throwing_distance=0.0
    if "Error"!=env_feedback:
        try:
            env_feedback_dict = parse_data(env_feedback)
            throwing_distance = env_feedback_dict["throwing_distance"]
            land_idx = env_feedback_dict["land_idx"]
            blouder_landing = env_feedback_dict["actual_position"][land_idx]
            startblock_landing = env_feedback_dict["startingBlock_position"][land_idx]
            startblock_blouder_dis_landing = np.linalg.norm(np.array(blouder_landing) - np.array(startblock_landing))
            if startblock_blouder_dis_landing<throwing_distance*0.25:
                not_dilvery=0.0
            max_height = env_feedback_dict["max_height"]
            if max_height<3.0:
                enough_height=0.0
            
            blouder_inital_height = env_feedback_dict["actual_position"][0][1]
            if blouder_inital_height<max_height:
                height_ratio += (max_height-blouder_inital_height)
            
            
            main_env_score = throwing_distance
            env_score = main_env_score
        except Exception as e:
            
            env_score=0
            pass

    score=not_dilvery*enough_height*(height_ratio*env_score)
    
    if score is None:  
        score = 0.0  
    if throwing_distance is None:  
        throwing_distance = 0.0 
    
    return {  
        "score": score,  
        "distance": throwing_distance  
    }
    
def car_scoring(env_feedback):
    score=0.0
    file_valid=1
    env_score=0.0
    throwing_distance=0
    if "Error"!=env_feedback:
        try:
            env_feedback_dict = parse_data(env_feedback)
            throwing_distance = env_feedback_dict["throwing_distance"]
            main_env_score = throwing_distance
            env_score = main_env_score
        except:
            env_score=0
            pass

    
    if score is None:  
        score = 0.0 
    if file_valid is None:  
        file_valid = 0.0 
    if throwing_distance is None:  
        throwing_distance = 0.0 
    if env_score is None:
        env_score=0.0
    score=file_valid*max(0,env_score)
    
    return {  
        "score": score,  
        "distance": throwing_distance
    }


def generate_random_string(length=8):
    random_string = secrets.token_hex(nbytes=length // 2)  
    return random_string

def read_txt(path):
    with open(path, "r", encoding="utf-8") as file:
        return file.read()

def write_file(path, content):
    with open(path, 'w', encoding='utf-8') as file:
        file.write(content)

def extract_json_from_string(input_string: str,return_raw_str=False):
    if isinstance(input_string,list):
        return input_string
    
    if os.path.exists(input_string):
        input_content = read_txt(input_string)
    else:
        input_content = deepcopy(input_string)
    

    match = re.search(r"```json(.*?)```", input_content, re.DOTALL)
    if match:
        json_content = match.group(1).strip()
        # print(json_content)
        try:
            if return_raw_str:
                return json.loads(json_content),json_content
            return json.loads(json_content)
        except json.JSONDecodeError as e:
            # print(e)
            pass 

    
    try:
        if return_raw_str:
            return json.loads(input_content),input_content
        return json.loads(input_content)
    except json.JSONDecodeError:
        try:
            if return_raw_str:
                return json.loads(input_content),input_content
            return ast.literal_eval(input_content)
        except (ValueError, SyntaxError):
            if return_raw_str:
                return None,""
            return None

def extract_json_from_string_extra(input_string: str,extra_index="json"):

    if os.path.exists(input_string):
        input_content = read_txt(input_string)
    else:
        input_content = deepcopy(input_string)
    
    
    match = re.search(fr"```{extra_index}(.*?)```", input_content, re.DOTALL)
    if match:
        json_content = match.group(1).strip()
        try:
            return json.loads(json_content)
        except json.JSONDecodeError:
            pass 

    
    try:
        return json.loads(input_content)
    except json.JSONDecodeError:
        try:
            return ast.literal_eval(input_content)
        except (ValueError, SyntaxError):
            return None

def match_case(input_str, case_l, case_r):

    escaped_case_l = re.escape(case_l)
    escaped_case_r = re.escape(case_r)
    
    pattern = fr'{escaped_case_l}(.*?){escaped_case_r}'
    
    matches = re.findall(pattern, input_str, re.DOTALL)
    
    return [match.strip() for match in matches if match.strip()]

def easy_get_bppos(blocktype, block_pos, block_rot, bp_id):
    if block_rot.shape != (4,):
        raise ValueError("Rotation must be a quaternion (4-element array).")
    
    rotation = R.from_quat(block_rot)
    bp_offset = BLOCKPROPERTY[str(blocktype)]["bc_bp"][bp_id]
    return block_pos + rotation.apply(bp_offset)

if __name__ == "__main__":
    pass