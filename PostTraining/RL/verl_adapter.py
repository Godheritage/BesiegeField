import json
import re
from AgenticCodes.utils import extract_json_from_string, write_file, easy_get_bppos,match_case
import AgenticCodes.data_processing as dp
from AgenticCodes.agentic_pipeline import AgenticPipeline
from AgenticCodes.config import DEFAULT_SAVE_ROOT
from concurrent.futures import ThreadPoolExecutor, as_completed,ProcessPoolExecutor
from typing import Any, List, Dict
from pathlib import Path
import os
import psutil
import ray
import threading
from verl.utils.logger.aggregate_logger import print_rank_0  
import time 
import numpy as np

TMP_PATH = Path(__file__).resolve().parent / "tmp"
os.makedirs(TMP_PATH,exist_ok=True)
MAX_WORKERS=64

def xvfb_99_is_running() -> bool:
    for proc in psutil.process_iter(['name', 'cmdline']):
        if proc.info['name'] == 'Xvfb':
            cmd = proc.info['cmdline'] or []
            if ':99' in cmd or f':99' in ' '.join(cmd):
                return True
    return False

current_ns = ray.get_runtime_context().namespace or None  
@ray.remote(name="GlobalAgenticPipelineSingleton",
            lifetime="detached",
            namespace=current_ns,
            max_concurrency=MAX_WORKERS)
class AgenticPipelineActor:
    def __init__(self,**kwargs):
        tasks_name = kwargs.get('tasks', "car/car_level1")
        self.agent = AgenticPipeline(
            save_root=DEFAULT_SAVE_ROOT,
            builder_bestofn=1,
            try_with3Dinfos=True,
            use_api=False,
            use_xvfb=xvfb_99_is_running(),
            tasks=[tasks_name] * MAX_WORKERS
        )
        self.agent._initialize_env_manager()
    def score(self, solution_str):
        return self.agent.process_response_auto(solution_str, 'single', {'tmp_path': TMP_PATH})

_actor_lock = threading.Lock()
_actor_initialized = False

def _ensure_actor_created(**kwargs):
    global _actor_initialized
    with _actor_lock:  
        if _actor_initialized:
            return
        try:
            AgenticPipelineActor.remote(**kwargs)
        except ValueError:
            pass
        _actor_initialized = True


def file_valid_check(solution_str):
    try:
        machines = extract_json_from_string(solution_str)
        if not is_list_of_dict(machines):
            return 0
        status, reason = dp.valid_check(machines, 1)
        return 1 if status != "Reject" and all(k not in reason for k in ("illegal building", "size error", "overlap error")) else 0
    except:
        return 0

def is_list_of_dict(obj: Any) -> bool:
    return isinstance(obj, list) and all(isinstance(item, dict) for item in obj)


def extract_win_condition(data: str) -> str | None:
    m = re.search(r'^Task type:\s*\{(.+?)\}\s*$', data, re.M)
    return m.group(1) if m else None

def extract_target_name(data: str) -> str | None:
    m = re.search(r'^Target name:\s*\{(.+?)\}\s*$', data, re.M)
    return m.group(1) if m else None


def parse_data(data):
    win_condition = extract_win_condition(data)
    if win_condition=="Boulder_throw":
        KEYWORD_MAPPING = {
            "Boulder moving distance": ("positive_distance", float),
            "Boulder max speed": ("max_speed", float),
            "Boulder max height": ("max_height", float),
            "Boulder average speed": ("average_speed", eval),
            "Boulder actual position": ("actual_position", eval),
            "startingBlock actual position": ("startingBlock_position", eval),
            "land_idx": ("land_idx", int),
        }
    elif win_condition=="Car_distance":
        KEYWORD_MAPPING = {
            "car moving distance": ("positive_distance", float),
            "car max speed": ("max_speed", float),
            "car max height": ("max_height", float),
            "car average speed": ("average_speed", eval),
        }
    elif win_condition=="Target_deliver":
        target_name = extract_target_name(data)
        moving_diatance = f"{target_name} moving distance"
        max_speed = f"{target_name} max speed"
        max_height = f"{target_name} max height"
        average_speed = f"{target_name} average speed"
        actual_position = f"{target_name} actual position"
        KEYWORD_MAPPING = {
            moving_diatance: ("positive_distance", float),
            max_speed: ("max_speed", float),
            max_height: ("max_height", float),
            average_speed: ("average_speed", eval),
            actual_position: ("actual_position", eval),
            "Game Win": ("game_win", bool)
        }
    elif win_condition=="Boulder_throw_with_target":
        KEYWORD_MAPPING = {
            "final distance between": ("negative_distance", float),
            "Boulder actual position": ("actual_position", eval),
            "Game Win": ("game_win", bool)
        }
    
    result = {key: False if key == "game_win" else None
          for key in {k[0] for k in KEYWORD_MAPPING.values()}}
    
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

def compute_score_batch(data_sources, solution_strs, ground_truths, extra_infos,**kwargs):
    win_condition = kwargs.get('win_condition', "Boulder_throw")
    score_handler = None
    
    if win_condition=="Car_distance":
        score_handler = compute_score_relu
    elif win_condition=="Boulder_throw":
        score_handler = compute_score_strict_throwtraj
    else:
        score_handler = compute_score_binary
    
    
    _ensure_actor_created(**kwargs)
    
    with ThreadPoolExecutor(max_workers=len(data_sources)) as executor:
        futures = []
        for data_source, solution_str, ground_truth, extra_info in zip(
            data_sources, solution_strs, ground_truths, extra_infos, strict=True
        ):
            future = executor.submit(score_handler, data_source, solution_str, ground_truth, extra_info)
            futures.append(future)

        results = [future.result() for future in futures]
    scores = [result["score"] for result in results]  
    valid_scores = [score for score in scores if score > 0]  
    valid_rate = len(valid_scores) / len(scores) if scores else 0.0 
    
    for result in results:  
        result["score_nonzero_rate"] = valid_rate
    
    scores = [result["file_valid"] for result in results]  
    valid_scores = [score for score in scores if score > 0]  
    valid_rate = len(valid_scores) / len(scores) if scores else 0.0 
    
    for result in results:  
        result["file_valid"] = valid_rate
    return results


def compute_score_binary(data_source, solution_str, ground_truth, extra_info=None):

    score=0.0
    file_valid=1
    
    file_valid = file_valid_check(solution_str)
    
    if file_valid:
        _agent_actor = ray.get_actor("GlobalAgenticPipelineSingleton", namespace=current_ns)
        kargs = ray.get(_agent_actor.score.remote(solution_str))
        if kargs:
            env_feedback = kargs["env_feedback"][0]
        else:
            env_feedback = "Error"

        if "Error"!=env_feedback:
            try:
                env_feedback_dict = parse_data(env_feedback)
                try:
                    score = env_feedback_dict["game_win"]
                except:
                    score=0
            except:
                score=0
                pass
        score=file_valid*max(0,score)
    
    return {  
        "score": score,   
        "file_valid":file_valid
    }


def compute_score_relu(data_source, solution_str, ground_truth, extra_info=None):

    score=0.0
    file_valid=1
    env_score=0.0
    positive_distance=0
    
    file_valid = file_valid_check(solution_str)
    
    if file_valid:
        _agent_actor = ray.get_actor("GlobalAgenticPipelineSingleton", namespace=current_ns)
        kargs = ray.get(_agent_actor.score.remote(solution_str))
        if kargs:
            env_feedback = kargs["env_feedback"][0]
        else:
            env_feedback = "Error"

        if "Error"!=env_feedback:
            try:
                env_feedback_dict = parse_data(env_feedback)
                try:
                    positive_distance = env_feedback_dict["positive_distance"]
                except:
                    positive_distance=0
                env_score = positive_distance
            except:
                env_score=0
                pass
        score=file_valid*max(0,env_score)
    
    return {  
        "score": score,  
        "distance": positive_distance,
        "file_valid":file_valid
    }

def compute_score_strict_throwtraj(data_source, solution_str, ground_truth, extra_info=None):

    score=0.0
    file_valid=1
    env_score=0
    positive_distance=0.0
    
    file_valid = file_valid_check(solution_str)
    
    if file_valid:
        _agent_actor = ray.get_actor("GlobalAgenticPipelineSingleton", namespace=current_ns)
        kargs = ray.get(_agent_actor.score.remote(solution_str))
        if kargs:
            env_feedback = kargs["env_feedback"][0]
        else:
            env_feedback = "Error"
        
        not_dilvery=1
        enough_height=1
        height_ratio=1.0
        if "Error"!=env_feedback:
            try:
                env_feedback_dict = parse_data(env_feedback)
                try:
                    positive_distance = env_feedback_dict["positive_distance"]
                except:
                    positive_distance=0
                land_idx = env_feedback_dict["land_idx"]
                blouder_landing = env_feedback_dict["actual_position"][land_idx]
                startblock_landing = env_feedback_dict["startingBlock_position"][land_idx]
                startblock_blouder_dis_landing = np.linalg.norm(np.array(blouder_landing) - np.array(startblock_landing))
                if startblock_blouder_dis_landing<positive_distance*0.25:
                    not_dilvery=0.0
                    
                max_height = env_feedback_dict["max_height"]
                if max_height<3.0:
                    enough_height=0.0

                blouder_inital_height = env_feedback_dict["actual_position"][0][1]
                if blouder_inital_height<max_height:
                    height_ratio += (max_height-blouder_inital_height)
                
                env_score = positive_distance
            except:
                env_score=0
                pass

        score=file_valid*not_dilvery*enough_height*(height_ratio*env_score)
    
    return {  
        "score": score,  
        "distance": positive_distance ,
        "file_valid":file_valid
    }

if __name__ == "__main__":
    pass