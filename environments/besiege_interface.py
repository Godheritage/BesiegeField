# besiege_interface.py
import os
from pickletools import string1
import time
import socket
import errno
import threading
import random, string
from .besiege_env import BesiegeEnv
from AgenticCodes.config import LEVEL_FILE_BASE
from environments.env_files.generate_random_level import generate_level_from_preset

def is_port_free(ip: str, port: int, *, udp=False) -> bool:
    """
    Check TCP/UDP port avaliable
    """
    sock_type = socket.SOCK_DGRAM if udp else socket.SOCK_STREAM
    with socket.socket(socket.AF_INET, sock_type) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            s.bind((ip, port))
            return True
        except OSError as e:
            # 98 = Address already in use
            if e.errno == errno.EADDRINUSE:
                return False
            raise

def alloc_port(start_port: int, max_retry: int = 100,
               *, ip: str = '0.0.0.0', udp=False) -> int:

    for offset in range(max_retry):
        port = start_port + offset
        if is_port_free(ip, port, udp=udp):
            return port
    raise RuntimeError(f'No free port found in range '
                       f'{start_port}-{start_port + max_retry - 1}')


def rand_name(length=8, suffix=''):
    chars = string.ascii_letters + string.digits
    return ''.join(random.choices(chars, k=length)) + suffix

def rand_level(level):
    try:
        level_full_path = os.path.join(LEVEL_FILE_BASE,"tmp_levels", rand_name('.blv'))
        generate_level_from_preset(preset_name=level,output_path=level_full_path)
        return level_full_path
    except:
        print("random level not supported, please check your level_presets.json, using original level")
        return None
    


class BesiegeEnvManager:
    """BesiegeField Pool"""
    def __init__(self, level_lists,use_xvfb):
        self.use_xvfb = use_xvfb
        self.env_dict, self.env_stat = self._initialize_envs(level_lists)
        self.env_lock = threading.Lock()

    def _initialize_envs(self, level_lists,random_in_init=False):
        env_dict = {}
        start_port = 6890
        for level in level_lists:
            start_port = alloc_port(start_port)
            besiege = BesiegeEnv(no_graphic=True, port=start_port, run_mode="costumelevelmode",use_xvfb = self.use_xvfb)
            
            if random_in_init:
                level_full_path = rand_level(level)
                if not level_full_path:
                    level_full_path = os.path.join(LEVEL_FILE_BASE, f"{level}.blv")
                    
            else:
                level_full_path = os.path.join(LEVEL_FILE_BASE, f"{level}.blv")
            
            load_command = f"LoadCustomLevel:{level_full_path}"
            
            print(f"PORT {start_port}: Loading custom level '{level}'...")
            besiege.send_message_waiting_receive(load_command, "CustomLevelLoaded")
            print(f"PORT {start_port}: Level '{level}' loaded.")
            
            env_dict.setdefault(level, []).append(besiege)
            start_port += 1
        
        env_stat = {env: False for env_list in env_dict.values() for env in env_list}
        return env_dict, env_stat

    def get_available_environment(self, task_name):
        """Get avaliable env"""
        while True:
            with self.env_lock:
                for env in self.env_dict.get(task_name, []):
                    if not self.env_stat[env]:
                        self.env_stat[env] = True
                        print(f"Environment {env.port} assigned to task '{task_name}'.")
                        return env
            print(f"No available environment for task '{task_name}', waiting...")
            time.sleep(5)

    def release_environment(self, env):
        """Release env"""
        with self.env_lock:
            if env in self.env_stat:
                self.env_stat[env] = False
                print(f"Environment {env.port} released.")
            else:
                print(f"Warning: Trying to release an untracked environment {env.port}.")
    
    def kill_all_processes(self):
        """kill env"""
        with self.env_lock:
            for env_list in self.env_dict.values():
                for env in env_list:
                    env.kill_process()

def besiege_level_menus(besiege_env: BesiegeEnv, bsgfile_path: str, instruction_list: list,random_per_simulate=False,level=None):
    """Run env and get feedback"""
    besiege_env.clear_receivequeue()
    if random_per_simulate:
        if not level:
            print("random need level name, please check besiege_level_menus function")
        else:
            level_full_path = rand_level(level)
            if level_full_path:
                load_command = f"LoadCustomLevel:{level_full_path}"
                besiege_env.send_message_waiting_receive(load_command, "CustomLevelLoaded")

    for instruct in instruction_list:
        # print(instruct)
        if "sleep" in instruct:
            _, duration = instruct.split(" ")
            time.sleep(int(duration))
        elif "LoadMachine:" in instruct:
            load_command = f"LoadMachine:{bsgfile_path}"
            besiege_env.send_message_waiting_receive(load_command, "MachineLoaded")
        elif "ToggleSimulate:_" in instruct:
            besiege_env.send_message_waiting_receive(instruct, "SimulationStart")
        elif "WaitingStop" in instruct:
            parts = instruct.split(" ")
            timeout = int(parts[1]) if len(parts) > 1 else None
            besiege_env.waiting_stop("SimulationEnd", timeout)
        elif "SwitchKey:" in instruct:
            besiege_env.send_message_waiting_receive(instruct, "PressKey")
    
    return besiege_env.get_receivequeue()
