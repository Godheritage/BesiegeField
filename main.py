import signal
import sys
import os
from pathlib import Path
import json
import threading
from environments.besiege_interface import BesiegeEnvManager
from AgenticCodes.agentic_pipeline import AgenticPipeline
from AgenticCodes.config import BASE_PATH,DEFAULT_SAVE_ROOT,APIPATH,LEVEL_FILE_BASE,SCRIPT_PATH,LEVELMENUS
import AgenticCodes.config as global_config

import argparse

    

exit_event = threading.Event()
def signal_handler(sig, frame):
    print("\nkilling process...")
    exit_event.set()
    sys.exit(0)


def override_global_config(args: argparse.Namespace, global_config) -> argparse.Namespace:
    for k, v in vars(args).items():
        if hasattr(global_config, k):
            setattr(global_config, k, v)

def run_agentic_pipeline(use_model, task, env_num, user_input,env_loop_run_times
                         ,continue_root,skip_designer,skip_inspector,skip_refiner,block_limitations):
    agentic_pipeline = AgenticPipeline(
        save_root=DEFAULT_SAVE_ROOT,
        model_name=use_model,
        tasks=[task] * env_num
    )

    agentic_pipeline.run(
        user_input=user_input,
        save=True,
        block_limitations=block_limitations,
        continue_root=continue_root,
        skip_designer=skip_designer,
        skip_inspector=skip_inspector,
        skip_refiner=skip_refiner,
        mcts_search_times=env_loop_run_times
    )
    agentic_pipeline._release_env_manager()
    
    

if __name__ == "__main__":    
    signal.signal(signal.SIGINT, signal_handler)  
    signal.signal(signal.SIGTERM, signal_handler) 

    parser = argparse.ArgumentParser(description="Run AgenticPipeline with specified parameters.")
    parser.add_argument("-use_model", type=str, default="gemini-2.5-pro", help="Model name to use")
    parser.add_argument("-task", type=str, default="catapult/catapult_level2", help="Task name")
    parser.add_argument("-env_num", type=int, default=2, help="Number of environments")
    parser.add_argument("-user_input", type=str, default="Design a machine to throw a boulder (type id 36) in a parabolic trajectory.", help="User input prompt")
    parser.add_argument("-env_loop_run_times", type=int, default=5, help="Running feedback-refine loop times")
    parser.add_argument("-continue_root", type=str, default=None, help="Root of previous experiment")
    parser.add_argument("-block_limitations", type=list, default=[0,1,2,5,9,15,16,22,30,35,36,41,63], help="block type range required the agent to use.")
    parser.add_argument("-skip_designer", type=bool, default=False, help="If skip designer (e.g. You have finished call designer in previous experiment)")
    parser.add_argument("-skip_inspector", type=bool, default=False, help="If skip inspector (e.g. You have finished call inspector in previous experiment)")
    parser.add_argument("-skip_refiner", type=bool, default=False, help="If skip refiner (e.g. You have finished call refiner in previous experiment)")
    parser.add_argument("-WHEEL_AUTO_ON", type=bool, default=True, help="If powered wheel auto working in game")
    
    parser.add_argument("-overwrite_levelmenu", type=bool, default=True, help="use task deafult config to overwrite prompt and block_limitations")
    
    args = parser.parse_args()
    override_global_config(args,global_config)
    
    if args.overwrite_levelmenu:
        args.user_input = LEVELMENUS[args.task]["deafult_prompt"]
        args.block_limitations = LEVELMENUS[args.task]["block_limitations"]
    
    run_agentic_pipeline(
        use_model=args.use_model,
        task=args.task,
        env_num=args.env_num,
        user_input=args.user_input,
        env_loop_run_times=args.env_loop_run_times,
        continue_root=args.continue_root,
        skip_designer=args.skip_designer,
        skip_inspector=args.skip_inspector,
        skip_refiner=args.skip_refiner,
        block_limitations=args.block_limitations
    )