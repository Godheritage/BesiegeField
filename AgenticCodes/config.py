# config.py
import json
from pathlib import Path
import traceback
import sys
from copy import deepcopy

# --- PATH SETTINGS ---
BASE_PATH = Path(__file__).resolve().parent
#These paths are probably you need to modify
APIPATH = BASE_PATH / "llm_call/chat_agent_api.json"
DEFAULT_SAVE_ROOT = BASE_PATH.parent /"AgenticResults"
SCRIPT_PATH = BASE_PATH.parent /"Besiege/run_besiegefield.sh"
#############################################

BLOCKPROPERTYPATH = BASE_PATH / "properties/besiege_blocks.json"
BLOCKINTROPATH = BASE_PATH / "properties/block_intro.json"
LEVEL_FILE_BASE = BASE_PATH.parent /"environments/env_files"
LEVELMENUSPATH = LEVEL_FILE_BASE / "level_menus.json"
# ---------------------


FORBIDEN_BLOCKS=[85,82,83,79,80,81,78,84]
WHEEL_AUTO_ON=True
LINEAR_BLOCKS=["7","9","45"]

TRAIN_INFERENCE_MAP = {
    "designer":[],
    "builder":["designer"],
    "quizzer":["designer","builder"],
    "modifier":["designer","builder","quizzer","env_querier"],
    "env_querier":["designer","builder","quizzer","modifier"],
}

# --- env config ---
LIFECYCLE = 0.2  # game atomic clock
MACHINE_TXT_FILENAMES = [
    'output.txt',
    "refined_machine.txt",
    "env_refined_machine",
    "final_machine"
]



INITJSON = """
```json
[{"id":"0","order_id":0,"parent":-1,"bp_id":-1}]
```
"""

def load_json_file(path):
    try:
        with open(path, 'r', encoding='utf-8') as file:
            return json.load(file)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        print(traceback.format_exc())
        return None

BLOCKPROPERTY = load_json_file(BLOCKPROPERTYPATH)
BLOCKINTRO = load_json_file(BLOCKINTROPATH)
LEVELMENUS = load_json_file(LEVELMENUSPATH)

FACINGMAP = {
"z+": {"Front": "z+", "Back": "z-", "Left": "x-", "Right": "x+", "Up": "y+", "Down": "y-"},
"z-": {"Front": "z-", "Back": "z+", "Left": "x+", "Right": "x-", "Up": "y+", "Down": "y-"},
"x-": {"Front": "x-", "Back": "x+", "Left": "z-", "Right": "z+", "Up": "y+", "Down": "y-"},
"x+": {"Front": "x+", "Back": "x-", "Left": "z+", "Right": "z-", "Up": "y+", "Down": "y-"},
"y+": {"Front": "y+", "Back": "y-", "Left": "x-", "Right": "x+", "Up": "z-", "Down": "z+"},
"y-": {"Front": "y-", "Back": "y+", "Left": "x-", "Right": "x+", "Up": "z+", "Down": "z-"}
}

    

if __name__ == "__main__":
    print("Base Path:", BASE_PATH)