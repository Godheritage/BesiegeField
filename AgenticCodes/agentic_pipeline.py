import os
import secrets
import shutil
import threading
import json
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import random
import math
import ast
import time
from scipy.spatial.transform import Rotation as R
import numpy as np
import copy
from copy import deepcopy
import traceback
from transformers import AutoTokenizer, AutoModelForCausalLM  
import torch
from collections import deque
import AgenticCodes.config as config
from AgenticCodes.utils import read_txt, write_file, extract_json_from_string, match_case,generate_random_string,blouder_throw_scoring,car_scoring
from environments.besiege_interface import BesiegeEnvManager, besiege_level_menus
import AgenticCodes.data_processing as dp 

from AgenticCodes.prompts import *
from AgenticCodes.llm_call.chat_agent import load_chat_agent, agent
from AgenticCodes.llm_call.text_refiner import TextRefiner
from AgenticCodes.config import APIPATH, LIFECYCLE,INITJSON,MACHINE_TXT_FILENAMES,LEVELMENUS
exp_list=[
"test-time-scaling",
"abl-short",
"abl-metadesigner",
"abl-detailed-designer",
"abl-env_querier",
"abl-quizzer",
"abl-modifier-history",
"abl-3dinfo",
"abl-auto-3dinfo",
"abl-random-search",
"abl-env-bon",
"abl-threed-rep",
"abl-self-critic"]

class LocalTextRefiner:
    def __init__(self,definition,temp_agent,temperature,top_p,return_handle):
        self.definition=definition
        self.temp_agent=temp_agent
        self.temperature=temperature
        self.top_p=top_p
        self.return_handle=return_handle

def quizzer_format_checking(quizzer_result):
    if "<Summary of Design Defects>" in quizzer_result and "</Summary of Design Defects>" in quizzer_result:
        conclusion = match_case(quizzer_result, "<Summary of Design Defects>", "</Summary of Design Defects>")[0]
        if "[" in conclusion and "]" in conclusion:
            return True
    return False

class AgenticPipeline:
    def __init__(self, model_name=None, save_root="", designer_n=8,
                 tasks=[],use_api=True,
                 designer_model_name=None,
                 builder_model_name = None,
                 modifyer_model_name = None,
                 env_querier_model_name = None,
                 quizzer_model_name = None,
                 use_xvfb = False,
                 model_path=None,
                 max_local=2):
        self.save_root = save_root or config.DEFAULT_SAVE_ROOT
        if model_name!=None:
            self.designer_model_name = model_name
            self.builder_model_name = model_name
            self.modifyer_model_name = model_name
            self.env_querier_model_name =model_name
            self.quizzer_model_name = model_name
        else:
            self.designer_model_name = designer_model_name
            self.builder_model_name = builder_model_name
            self.modifyer_model_name = modifyer_model_name
            self.env_querier_model_name =env_querier_model_name
            self.quizzer_model_name = quizzer_model_name
        
        self.designer_n = designer_n
        self.use_api = use_api
        self.exp_type = None
        self.custome_designer_output=None
        self.model_path = model_path
        
        self.max_local=max_local
        self.local_agents=[]
        self._idle_indices= deque()     
        self._in_use      = set()       
        self._cond        = threading.Condition()  
        
        self.tasks = tasks or (["car/car_level1"] * 5)
        self.env_manager = None 
        
        
        self.designer_save_root = None
        self.builder_save_root = None
        self.markov_save_root = None
        self.designer_agent = None
        self.builders = []
        self.designer_structures = []
        self.designer_block_limitations = []
        
        self.use_xvfb = use_xvfb

    def _initialize_env_manager(self):
        if self.env_manager is None:
            print("Initializing Besiege Environment Manager...")
            self.env_manager = BesiegeEnvManager(self.tasks,self.use_xvfb)
    
    def _release_env_manager(self):
        if hasattr(self, 'env_manager') and self.env_manager is not None:
            print("Clearing envs")
            self.env_manager.kill_all_processes()
        else:
            pass

    def _update_used_model(self,model_name):
        self.designer_model_name = model_name
        self.builder_model_name = model_name
        self.modifyer_model_name = model_name
        self.env_querier_model_name =model_name
        self.quizzer_model_name = model_name

    def prepare_agent_auto(self,temperature=0.7,top_p=0.95,
                           agent=""
                           ,custome_prompt = "",custome_model_name = "deepseek-chat"
                           ,block_limitations=None,builder_nums=1,metadesigner_prompt_type = "deafult"
                           ,designer_prompt_type= "short",
                           inspector_prompt_type = "short",
                           refiner_prompt_type = "short",
                           env_querier_prompt_type = "short",
                           need_prompt = False):
        
        if agent=="custome":
            model_name = custome_model_name
            agent_num = 1
            definition = custome_prompt
            samples=[]
            historical_num=0
        else:
            model_name = self.builder_model_name
            if agent=="designer":
                agent_num = builder_nums
            else: agent_num =1
            
            agent_prompt_type_map={
                "meta-designer":metadesigner_prompt_type,
                "designer":designer_prompt_type,
                "inspector":inspector_prompt_type,
                "refiner":refiner_prompt_type,
                "env_querier":env_querier_prompt_type,
            }
            
            definition,samples=format_agentic_systemprompt(agent,agent_prompt_type_map[agent],block_limitations)
            historical_num=0
            
        model_name ,api_version, CoT = load_chat_agent(APIPATH,model_name=model_name)

        auto_agents=[]
        for i in range(agent_num):
            if self.use_api:
                auto_agent = TextRefiner(
                    definition=definition,
                    model_name=model_name,
                    CoT=CoT,
                    samples=samples,
                    temperature=temperature,
                    top_p=top_p,
                    historical_num=historical_num,
                    api_version=api_version
                )
            else:
                with self._cond:
                    need = self.max_local - len(self.local_agents)
                    for _ in range(need):
                        agent = AutoModelForCausalLM.from_pretrained(
                            self.model_path,
                            torch_dtype=torch.bfloat16,
                            device_map="auto",
                            trust_remote_code=True
                        )
                        idx = len(self.local_agents)
                        self.local_agents.append(agent)
                        self._idle_indices.append(idx)   
                auto_agent= LocalTextRefiner(definition=definition,temp_agent=None,temperature=temperature,top_p=top_p,return_handle=None)
            auto_agents.append(auto_agent)
        if need_prompt:
            return auto_agents,definition
        else:
            return auto_agents

    def take_one_agent(self, local_agent):
        with self._cond:
            while not self._idle_indices:
                self._cond.wait()
            idx = self._idle_indices.popleft()   
            self._in_use.add(idx)                
            raw_agent = self.local_agents[idx]   
            local_agent.temp_agent = raw_agent
            local_agent.return_handle = idx
            
    def return_agent(self, local_agent):
        with self._cond:
            idx = local_agent.return_handle
            self._in_use.remove(idx)
            self._idle_indices.append(idx)
            local_agent.temp_agent = None
            local_agent.return_handle =None
            self._cond.notify()      

    def chat_once(self,role_agent,input,max_retry=5):
        if self.use_api:
            retry=0
            result=None
            result = agent(role_agent,input)
            if result is None and retry<max_retry:
                result = agent(role_agent,input)
                retry+=1
            if result==None:
                print("Error, call LLM failed")
        else:
            retry=0
            result=None
            self.take_one_agent(role_agent)
            result = self.chat(
                tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True),
                model = role_agent.temp_agent,
                max_new_tokens=4096,
                temperature = role_agent.temperature,
                top_p=role_agent.top_p,
                system_prompt=role_agent.definition,
                user_prompt=input
            )
            while result is None and retry<max_retry:
                result = self.chat(
                    tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True),
                    model = role_agent.temp_agent,
                    max_new_tokens=4096,
                    temperature = role_agent.temperature,
                    top_p=role_agent.top_p,
                    system_prompt=role_agent.definition,
                    user_prompt=input
                )
                retry+=1
            if result==None:
                print("Error, call LLM failed")
            self.return_agent(role_agent)
        return result

    @torch.inference_mode()
    def chat(self,tokenizer, model, max_new_tokens: int = 512, device: str = "cpu",do_sample=True,temperature=0.7,system_prompt=None,user_prompt=None,top_p=0.95):
        messages = [{"role": "system", "content": system_prompt},{"role": "user", "content": user_prompt}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(model.device)
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature,
            top_p=top_p,
            pad_token_id=tokenizer.eos_token_id
        )
        new_tokens = output_ids[0][len(inputs.input_ids[0]):]
        content= tokenizer.decode(new_tokens, skip_special_tokens=True)
        return {"content": content,"history":messages}

    def ask_metadesigner(self, user_input,metadesigner_agent, save=True, debug=False):
        output_save_path = os.path.join(self.agentic_paths["meta-designer"], "output.txt")
        input_save_path = os.path.join(self.agentic_paths["meta-designer"], "input.txt")
        agent_translator = metadesigner_agent

        out_contents = self.chat_once(agent_translator, user_input)
        history = out_contents["history"][0]["content"]
        if save:
            os.makedirs(os.path.dirname(output_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(input_save_path), exist_ok=True)
            write_file(output_save_path, out_contents["content"])
            write_file(input_save_path, f"system:\n{history}\nuser:\n{user_input}")
            if "reasoning_content" in out_contents:
                CoT_save_path = os.path.join(self.agentic_paths["meta-designer"], "CoT.txt")
                os.makedirs(os.path.dirname(CoT_save_path), exist_ok=True)
                write_file(CoT_save_path, out_contents["reasoning_content"])
        if debug:
            for k, v in out_contents.items():
                print(f"designer:\n{k}:\n{v}")

        return out_contents["content"]

    def ask_designer(self, designer_agents,designer_id, structurechain, user_input, save=True, debug=False):
        designer_path = os.path.join(
            self.agentic_paths["designer"],
            f"designer_{designer_id}",
            f"structurechain_{structurechain}"
        )
        output_save_path = os.path.join(designer_path, "output.txt")
        input_save_path = os.path.join(designer_path, "input.txt")
        CoT_save_path = os.path.join(designer_path, "CoT.txt")
        agent_translator = designer_agents[int(designer_id)]
        out_contents = self.chat_once(agent_translator, user_input)
        history = out_contents["history"][0]["content"]
        if save:
            os.makedirs(os.path.dirname(output_save_path), exist_ok=True)
            os.makedirs(os.path.dirname(input_save_path), exist_ok=True)
            write_file(output_save_path, out_contents["content"])
            write_file(input_save_path, f"system:\n{history}\nuser:\n{user_input}")
            if "reasoning_content" in out_contents:
                os.makedirs(os.path.dirname(CoT_save_path), exist_ok=True)
                write_file(CoT_save_path, out_contents["reasoning_content"])
        if debug:
            for k, v in out_contents.items():
                print(f"builder_{designer_path}_{structurechain}:\n{k}:\n{v}")

        return out_contents["content"]
    
    
    def process_metadesigner_output(self,output=None,debug_path =None):
        if debug_path:
            output = read_txt(debug_path)

        designer_output = extract_json_from_string(output)

        conclusion_list = []
        block_limitation_list = []
        required_keys = {"build_order", "design_structure", "machine_structure"}
        if not all(
            isinstance(designer_output.get(key), (dict, list)) and designer_output[key]
            for key in required_keys
        ):
            raise ValueError("Some necessary key non-exist, type error or none")
        
        build_order = designer_output.get("build_order", [])
        design_structure = designer_output.get("design_structure", [])
        machine_structure = designer_output.get("machine_structure", {})
        for i, structure in enumerate(build_order):
            for design in design_structure:
                if structure == design["function_name"]:
                    function_name = design["function_name"]
                    description = design["description"]
                    related_function_points = design["related_function_points"]
                    block_id = machine_structure[function_name]["block_id"]
                    reason = machine_structure[function_name]["reason"]

                    head = "On the basis of the existing structure, generate"
                    middle = "as"
                    use_block_type = "using block type"
                    conclusion = f"{head}[{function_name}],[{description}],{middle}{related_function_points},{use_block_type}{block_id},[{reason}]"
                    
                    conclusion_list.append(conclusion)
                    block_limitation_list.append(block_id)
        
        self.designer_structures = conclusion_list
        self.designer_block_limitations = block_limitation_list
        return conclusion_list,block_limitation_list


    def structure_valid_check(self, structures_currentlevel, current_structurechain,delete=False,level="designer"):
        valid_outputs = []
        valid_chain = []

        for structure, chain in zip(structures_currentlevel, current_structurechain):
            designer_id = chain.split("-")[-1]
            
            check_result,error_reason = dp.valid_check(structure,return_error_reason= True)
            if check_result is True:
                valid_outputs.append(structure)
                valid_chain.append(chain)
                continue 
            
            designer_folder_name = f"designer_{designer_id}"
            structure_folder_name = f"structurechain_{chain}"
            fail_folder = os.path.join(self.agentic_paths["designer"], designer_folder_name, structure_folder_name)
            if level=="designer":
                fail_file="output.txt"
            else:
                raise KeyError("no avaliable files")

            if delete:
                if check_result is False:
                    if os.path.exists(fail_folder):
                        try:
                            shutil.rmtree(fail_folder)
                        except OSError as e:
                            print(f"deleting {fail_folder} error: {e}")
                    else:
                        print("error file not found")
                        print(fail_folder)
                
                elif check_result == "Reject":
                    print(f"{fail_folder} designer output format reject, there may some error inside")
            else:
                if "illegal building:" in error_reason:
                    rename_file = "output_file_error.txt"
                if "size error:" in error_reason or "overlap error:" in error_reason:
                    rename_file = "output_threed_error.txt"
                if "illegal attachable face" in error_reason:
                    rename_file = "output_file_error.txt"
                try:
                    fail_folder = Path(fail_folder)  
                    old_path = fail_folder / fail_file 
                    new_path = fail_folder / rename_file
                    old_path.rename(new_path)
                except OSError as e:
                    print(f"renaming path {old_path} error: {e}")
                    

        return valid_outputs, valid_chain

    def save_machine(self):
        for dirpath, dirnames, filenames in os.walk(self.agentic_paths["save_root"]):
            for filename in filenames:
                for machine_json_name in MACHINE_TXT_FILENAMES:
                    if machine_json_name in filename.lower() and ".bsg" not in filename.lower():
                        save_name = filename.lower().replace(".txt",".bsg")
                        if self.exp_type=="abl-threed-rep":
                            xml_save_path = os.path.join(dirpath,save_name)
                            filepath = os.path.join(dirpath, filename)
                            with open(filepath, 'r', encoding='utf-8') as f:
                                test_3drep = f.read()
                            st_tree = dp.abl_3djson_to_treejson(test_3drep)
                            pattern = r'(?s)```json.*?```'
                            new_st = re.sub(pattern,st_tree,test_3drep,count=1)
                            dp.json_to_xml(new_st,xml_save_path)
                        else:
                            try:
                                xml_save_path = os.path.join(dirpath,save_name)
                                filepath = os.path.join(dirpath, filename)
                                dp.json_to_xml(filepath,xml_save_path)
                            except:
                                pass
                    

    def call_metadesigner(self, user_input, save, continue_root, block_limitations=None):
        if block_limitations is None:
            block_limitations = []

        metadesigner_output = None
        if continue_root:
            metadesigner_save_root = self.agentic_paths["meta-designer"]
            output_path = os.path.join(metadesigner_save_root, "output.txt")
            if os.path.exists(output_path):
                metadesigner_output = read_txt(output_path)
            else:
                for file_name in os.listdir(metadesigner_save_root):
                    if "output" in file_name:
                        output_path = os.path.join(metadesigner_save_root, file_name)
                        metadesigner_output = read_txt(output_path)
                        break
            self.agentic_paths["meta-designer-details"]={}
            self.agentic_paths["meta-designer-details"]["input"]=output_path.replace("output.txt","input.txt")
            self.agentic_paths["meta-designer-details"]["output"]=output_path


        # Ensure directories exist
        os.makedirs(self.agentic_paths["meta-designer"], exist_ok=True)


        # 2. Get metadesigner_output (if not already loaded from file)
        if metadesigner_output is None:
            if self.exp_type == "abl-detailed-designer":
                metadesigner_prompt_type = "ablation"
            else:
                metadesigner_prompt_type = "deafult"
            metadesigner_agent = self.prepare_agent_auto(
                temperature=0.7, agent="meta-designer",
                metadesigner_prompt_type=metadesigner_prompt_type
            )[0]
            
            if block_limitations:
                additional_blocks_str = f'{"The following are the recommended block types for building the machine. Please flexibly select the required block types based on the recommendations and other block features."}{block_limitations}'
                user_input += "\n" + additional_blocks_str

            metadesigner_output = self.ask_metadesigner(user_input,metadesigner_agent,save=save)

        # 3. Process output and return
        conclusion_list, block_limitation_list = self.process_metadesigner_output(metadesigner_output)

        return {
            "designer_output": metadesigner_output,
            "conclusion_list": conclusion_list,
            "block_limitation_list": block_limitation_list
        }
        
    def abl_call_designer(self, user_input, block_limitations=None):
        if block_limitations is None: 
            block_limitations = []
        os.makedirs(self.agentic_paths["designer"], exist_ok=True)
        valid_outputs = []
        valid_chain = []
        designer_agents = self.prepare_agent_auto(
                temperature=0.7, agent="designer",
                block_limitations=block_limitations, builder_nums=self.designer_n,
                designer_prompt_type="ablation_singleagent"
            )
        current_inputs, current_structurechain = self.prepare_designer_inputs(
            user_input, True, valid_outputs, valid_chain
        )
        valid_outputs, valid_chain = self.parallel_run_designer(
            current_inputs=current_inputs, current_structurechain=current_structurechain, debug=True,max_retry_time=5,designer_agents=designer_agents
        )
        self.save_machine()
        

    
    def call_designer(self, skip_designer=False, save=True, debug=False):
        if skip_designer:
            return self._load_designer_results_from_disk()

        valid_outputs = []
        valid_chain = []
        
        num_structures = len(self.designer_structures)

        for i in range(num_structures):
            is_initial_build = (i == 0)
            structure_instruction = self.designer_structures[i]
            block_limitation = list(dict.fromkeys(
                item
                for j in range(i+1)
                for item in self.designer_block_limitations[j]
            ))

            if debug:
                print(f"--- start designer {i+1}/{num_structures} ---")
                print(f"prompt: {structure_instruction[:50]}...")
                print(f"block limitations: {block_limitation}")

            prompt_type = "short"
            
            if self.exp_type=="abl-threed-rep":
                prompt_type="short_abl_3drep"
            
            
            designer_agents = self.prepare_agent_auto(
                temperature=0.7, agent="designer",
                block_limitations=block_limitation, builder_nums=self.designer_n,
                designer_prompt_type=prompt_type
            )

            current_inputs, current_structurechain = self.prepare_designer_inputs(
                structure_instruction, is_initial_build, valid_outputs, valid_chain
            )

            valid_outputs, valid_chain = self.parallel_run_designer(
                current_inputs=current_inputs, current_structurechain=current_structurechain, debug=debug,max_retry_time=5,designer_agents=designer_agents
            )
            
            if debug:
                print(f"designer round {i+1} finish, valid: {len(valid_outputs)}")
            
            if not valid_outputs:
                print(f"warning: designer round {i+1} all failed")
                return [], []

        if save:
            self.save_machine()
        
        self.agentic_paths["structure_chain"]=[]
        for chain in valid_chain:
            self.agentic_paths["structure_chain"].append(f"structurechain_{chain}")
        return valid_outputs, valid_chain

    def _load_designer_results_from_disk(self):
        max_len = -1
        final_chains = []
        final_outputs = []

        for root, dirs, _ in os.walk(self.agentic_paths["designer"]):
            for dir_name in dirs:
                if "structure" not in dir_name:
                    continue
                try:
                    chain_str = dir_name.split("_")[1]
                    current_len = len(chain_str.split("-"))
                except IndexError:
                    continue

                output_path = os.path.join(root, dir_name, "output.txt")
                if not os.path.exists(output_path):
                    continue

                output_content = read_txt(output_path)

                if current_len > max_len:
                    max_len = current_len
                    final_chains = [chain_str]
                    final_outputs = [output_content]
                elif current_len == max_len:
                    final_chains.append(chain_str)
                    final_outputs.append(output_content)
        self.agentic_paths["structure_chain"]=[]
        if not final_outputs:
            return [], []
        valid_outputs,valid_chain = self.structure_valid_check(final_outputs, final_chains)
        for chain in valid_chain:
            self.agentic_paths["structure_chain"].append(f"structurechain_{chain}")
        return valid_outputs,valid_chain


    def prepare_designer_inputs(self, structure_instruction, init_prompt, valid_outputs, valid_chain):
        if init_prompt:
            if self.exp_type=="abl-threed-rep":
                init_json = """```json[{"GP":[0,5.05,0],"GR":[0,0,0,1]},{"id": 0, "Position":[0,0,0], "Rotation":[0,0,0,1]}]```"""
            else:
                init_json = INITJSON
            
            user_input = f"{structure_instruction}\n{init_json}"
            current_inputs = [user_input] * self.designer_n
            current_structurechain = [str(i) for i in range(self.designer_n)]
            return current_inputs, current_structurechain

        source_data = []
        for i, out_str in enumerate(valid_outputs):
            try:
                out_json = extract_json_from_string(out_str)
                threeD_info = dp.get_3Dinfos_from_json(out_json)
                threeD_info_input = json.dumps(threeD_info, ensure_ascii=False)
                if self.exp_type =="abl-3dinfo":
                    threeD_info_input=""
                    
                source_data.append({
                    "json_str": json.dumps(out_json, ensure_ascii=False),
                    "3d_info_str": threeD_info_input,
                    "chain": valid_chain[i]
                })
            except Exception as e:
                print(f"Error processing (chain: {valid_chain[i]}): {e}")
        
        if not source_data:
            print('no source_data')
            return [], []

        current_inputs = []
        current_structurechain = []
        num_sources = len(source_data)

        for j in range(self.designer_n):
            source = source_data[j % num_sources]
            user_input = (
                f"{structure_instruction}\n"
                f"{source['json_str']}\n"
                f"{source['3d_info_str']}"
            )
            new_chain = f"{source['chain']}-{j}"
            current_inputs.append(user_input)
            current_structurechain.append(new_chain)
        return current_inputs, current_structurechain

    def parallel_run_designer(self, current_inputs, current_structurechain, max_retry_time=5, save=True, debug=False,designer_agents=None):
        while max_retry_time >= 0:
            max_retry_time -= 1
            structures_currentlevel = [None] * self.designer_n
            with ThreadPoolExecutor(max_workers=self.designer_n) as executor:
                futures = {
                    executor.submit(
                        self.ask_designer,
                        designer_id=j,
                        structurechain=current_structurechain[j],
                        user_input=current_inputs[j],
                        save=save,
                        debug=False,designer_agents=designer_agents
                    ): j for j in range(self.designer_n)
                }
                for fut in as_completed(futures): 
                    idx = futures[fut]          
                    if not fut.exception():
                        structures_currentlevel[idx] = fut.result()
                    else:
                        structures_currentlevel[idx] = None   

            if self.exp_type=="abl-threed-rep":
                structures_currentlevel_new = []
                for st in structures_currentlevel:
                    try:
                        st_tree = dp.abl_3djson_to_treejson(st)
                        pattern = r'(?s)```json.*?```'
                        new_st = re.sub(pattern,st_tree,st,count=1)
                        structures_currentlevel_new.append(new_st)
                    except:
                        print("threed to tree failed")
                        structures_currentlevel_new.append(None)
                structures_currentlevel = structures_currentlevel_new
                
            
            valid_outputs, valid_chain = self.structure_valid_check(structures_currentlevel, current_structurechain)
            
            if debug:
                print(f"{len(structures_currentlevel)} designer output")
                print(f"valid designer machine{len(valid_outputs)}")
                print(f"valid chain{valid_chain}")
                
            if valid_outputs:
                return self.structure_valid_check(valid_outputs, valid_chain)
        
        return [], []

    def lazy_update(self,model_name):
        self.designer_model_name = model_name
        self.builder_model_name = model_name
        self.modifyer_model_name = model_name
        self.env_querier_model_name =model_name
        self.quizzer_model_name = model_name
    
    def _auto_get_path(self,continue_root=None):
        agent_paths = {}
        if continue_root:
            agent_paths["save_root"]=continue_root
        else:
            current_time = datetime.now().strftime("%Y-%m-%d-%H-%M")
            markov_save_root = os.path.join(self.save_root, current_time)
            agent_paths["save_root"]=markov_save_root
        
        agent_names = ["meta-designer","designer","inspector","refiner","env_loop"]
        for agent in agent_names:
            model_name = self.designer_model_name.replace("/","_")
            agent_folder = f"{agent}_{model_name}"
            agent_paths[agent] = os.path.join(agent_paths["save_root"], agent_folder)
        return agent_paths
            

    def run(self,user_input,save,continue_root=None,skip_designer = False,skip_inspector=False,
            skip_refiner=False,block_limitations=[],no_envloop=False,mcts_search_times=5):

        self.agentic_paths = self._auto_get_path(continue_root=continue_root)
        if self.exp_type=="abl-metadesigner":
            self.abl_call_designer(user_input,block_limitations=block_limitations)
            return
        else:
            if not self.custome_designer_output:
                designer_output = self.call_metadesigner(user_input,save,continue_root,block_limitations=block_limitations)
            else:
                designer_output = {
                    "designer_output": self.custome_designer_output,
                    "conclusion_list": [],
                    "block_limitation_list": block_limitations,
                }
            
            valid_outputs,valid_chain = self.call_designer(skip_designer = skip_designer,debug=True)
            if valid_outputs==[] and valid_chain==[]:
                print("error! designer no valid results")
                return
        
        if self.exp_type=="abl-detailed-designer" or self.exp_type=="abl-threed-rep":
            return
        
        quizes,quiz_inputs,save_paths,threeDinfos,machine_jsons = self.call_inspector(valid_chain,skip_inspector=skip_inspector
                                                                                    ,block_limitations=block_limitations)
        
        modifyer_inputs,modify_results,modified_machine_jsons,refiner_save_paths = self.call_refiner(designer_output,
                                                                                   quizes,save_paths,
                                                                                   threeDinfos,machine_jsons,
                                                                                   skip_refiner=skip_refiner,
                                                                                   block_limitations=block_limitations)
        #############
        if self.exp_type=="abl-3dinfo" or self.exp_type=="abl-auto-3dinfo":
            return

        #############env simulate
        if no_envloop:
            return
        
        search_strategy="MCTS"
        if self.exp_type=="abl-random-search":
            search_strategy="random"
        if self.exp_type=="abl-env-bon":
            search_strategy="bon"
        
        self.env_modify_loop(modified_machine_bsg_paths=refiner_save_paths,
                                designer_output=designer_output,
                                mcts_search_times=mcts_search_times,
                                search_strategy=search_strategy,
                                block_limitations=block_limitations)
        #############
    
    def get_required_list(self,modifier_result):
        required_feedback = []
        for i, result in enumerate(modifier_result):
            try:
                required_feedback_str = match_case(result,"<Required Feedback>","</Required Feedback>")[0]
                required_feedback_dict = json.loads(required_feedback_str)
                assert(isinstance(required_feedback_dict, list) and len(required_feedback_dict) > 0)
                formated_required_feedback_dict=[]
                for block_info in required_feedback_dict:
                    try:
                        order_id = int(block_info["order_id"])
                        duration = block_info["duration"]
                        assert(isinstance(duration, list) and len(duration) == 2)
                        duration_s = float(duration[0])
                        duration_e = float(duration[1])
                        properties = block_info["properties"]
                        assert(isinstance(properties, list))
                        for prop in properties:
                            assert(prop in ["position", "rotation", "velocity","length"])
                        formated_required_feedback_dict.append(block_info)
                    except:
                        continue
                required_feedback.append(formated_required_feedback_dict)
            except:
                required_feedback.append([])
        return required_feedback

    def best_of_n_env_loop(self,modified_machine_bsg,modified_machine_output, save_path, designer_output, task, threeDinfo=None, max_loop=5, 
                    block_limitations=[],init_machine_result=None):
        def calculate_best(children_ids, mct_tree):
            best_ucb_value = -float('inf')
            best_child_node = None
            for child_id in children_ids:
                child_node = mct_tree[child_id]
                ucb_value = child_node["sim_value"]
                if best_ucb_value<ucb_value:
                    best_ucb_value=ucb_value
                    best_child_node = child_node
            return best_child_node
        
        def get_best_n(mct_tree):
            id2node = {n["node_id"]: n for n in mct_tree}

            depth = {}
            def get_depth(node_id):
                if node_id is None or node_id==-1:               
                    return -1
                if node_id in depth:               
                    return depth[node_id]
 
                d = get_depth(id2node[node_id]["parent_id"]) + 1
                depth[node_id] = d
                return d

 
            max_d = -1
            for n in mct_tree:
                d = get_depth(n["node_id"])
                if d > max_d:
                    max_d = d

  
            bon_childs =  [n for n in mct_tree if get_depth(n["node_id"]) == max_d] 
            
            best_node=None
            best_score=-100
            for node in bon_childs:
                if node["sim_value"]>best_score:
                    best_node=node
                    best_score=node["sim_value"]
            return best_node
        mct_tree = []
        init_required = None

        init_machine_result = modified_machine_output
        if init_machine_result is not None:
            init_machine_result_str = read_file(init_machine_result)
            init_required = self.get_required_list([init_machine_result_str])[0]

        mcts_save_root = os.path.join(save_path, f"BoN_loop")
        os.makedirs(mcts_save_root, exist_ok=True)
        root = self.init_node(modified_machine_bsg=modified_machine_bsg,
                            mcts_save_root=mcts_save_root,
                            designer_output = designer_output,
                            block_limitations=block_limitations,
                            simulate_loop="0",
                            required_feedback=init_required,task=task)
        root["node_id"] = 0
        root["parent_id"] = -1
        mct_tree.append(root)
        
        for j in range(max_loop):
            current_node = get_best_n(mct_tree)
            _,modifier_result,modified_machine_json, machine_bsg = self.call_env_refiner(designer_output=designer_output,
                                                machine_json=current_node["machine_txt"],
                                                threeDinfo=threeDinfo,
                                                env_feedback=current_node["env_feedback"],
                                                save=True,
                                                save_path=mcts_save_root,
                                                block_limitations=block_limitations,
                                                start_mcts_id=len(mct_tree)
                                                )
            if machine_bsg =="Error":
                print("Error: BoN expand error")
            else:
                for i, child_machine_bsg in enumerate(machine_bsg):
           
                    mct_tree_length = len(mct_tree)
                    child_node = self.init_node(modified_machine_bsg=child_machine_bsg,
                                            mcts_save_root=mcts_save_root,
                                            simulate_loop=f"{mct_tree_length}",
                                            designer_output = designer_output,
                                            block_limitations=block_limitations,task=task)
                    child_node["node_id"] = mct_tree_length
                    mct_tree.append(child_node)
                    child_node["parent_id"] = current_node["node_id"]
                    current_node["children_ids"].append(child_node["node_id"])
                    
        best_act = calculate_best(list(range(0, len(mct_tree))), mct_tree)
        best_id = best_act["node_id"]
        with open(os.path.join(mcts_save_root, f"bon_best_{best_id}.json"), "w", encoding="utf-8") as f:
            json.dump(mct_tree, f, ensure_ascii=False, indent=4)
        
        modified_machine_bsg=best_act["machine_bsg"]
        init_machine_result = best_act["modifier_result"]

        shutil.copy(modified_machine_bsg, os.path.join(save_path, f"final_machine_bon.bsg"))
        base_path = os.path.splitext(modified_machine_bsg)[0]
        modified_machine_txt = base_path + ".txt"
        shutil.copy(modified_machine_txt, os.path.join(save_path, f"final_machine_bon.txt"))
        shutil.copy(init_machine_result, os.path.join(save_path, f"output_bon.txt"))
        print(f"final machine:{modified_machine_bsg}")
        
        return best_act["machine_bsg"],best_act["modifier_result"]

    def single_loop(self,modified_machine_bsg,modified_machine_output, save_path, designer_output, task, threeDinfo=None, max_loop=5, 
                    block_limitations=[],init_machine_result=None):
        def calculate_best(children_ids, mct_tree,skip_nonvisited=False):
            best_ucb_value = -float('inf')
            best_child_node = None
            for child_id in children_ids:
                child_node = mct_tree[child_id]
                ucb_value = child_node["sim_value"]
                if best_ucb_value<ucb_value:
                    best_ucb_value=ucb_value
                    best_child_node = child_node
            return best_child_node
        
        mct_tree = []
        init_required = None

        init_machine_result = modified_machine_output
        if init_machine_result is not None:
            init_machine_result_str = read_file(init_machine_result)
            init_required = self.get_required_list([init_machine_result_str])[0]

        mcts_save_root = os.path.join(save_path, f"random_loop")
        os.makedirs(mcts_save_root, exist_ok=True)
        root = self.init_node(modified_machine_bsg=modified_machine_bsg,
                            mcts_save_root=mcts_save_root,
                            designer_output = designer_output,
                            block_limitations=block_limitations,
                            simulate_loop=f"0",
                            required_feedback=init_required,task=task)
        root["node_id"] = 0
        root["parent_id"] = -1
        mct_tree.append(root)
        
        for j in range(max_loop):
            current_node = mct_tree[-1] 
            _,modifier_result,modified_machine_json, machine_bsg = self.call_env_refiner(designer_output=designer_output,
                                                    machine_json=current_node["machine_txt"],
                                                    threeDinfo=threeDinfo,
                                                    env_feedback=current_node["env_feedback"],
                                                    save=True,
                                                    save_path=mcts_save_root,
                                                    block_limitations=block_limitations,
                                                    start_mcts_id=len(mct_tree),
                                                    parallel_num=1
                                                    )
            if machine_bsg =="Error":
                print("MCTS expansion but get no new child")
            else:
                for i, child_machine_bsg in enumerate(machine_bsg):

                    mct_tree_length = len(mct_tree)
                    child_node = self.init_node(modified_machine_bsg=child_machine_bsg,
                                            mcts_save_root=mcts_save_root,
                                            simulate_loop=f"{mct_tree_length}",
                                            designer_output = designer_output,
                                            block_limitations=block_limitations,task=task)
                    child_node["node_id"] = mct_tree_length
                    mct_tree.append(child_node)
                    child_node["parent_id"] = current_node["node_id"]
                    current_node["children_ids"].append(child_node["node_id"])
                    
        best_act = calculate_best(list(range(0, len(mct_tree))), mct_tree,skip_nonvisited=True)
        best_id = best_act["node_id"]
        with open(os.path.join(mcts_save_root, f"random_best_{best_id}.json"), "w", encoding="utf-8") as f:
            json.dump(mct_tree, f, ensure_ascii=False, indent=4)
        
        modified_machine_bsg=best_act["machine_bsg"]
        init_machine_result = best_act["modifier_result"]

        shutil.copy(modified_machine_bsg, os.path.join(save_path, f"final_machine_random.bsg"))
        base_path = os.path.splitext(modified_machine_bsg)[0]
        modified_machine_txt = base_path + ".txt"
        shutil.copy(modified_machine_txt, os.path.join(save_path, f"final_machine_random.txt"))
        shutil.copy(init_machine_result, os.path.join(save_path, f"output_random.txt"))
        print(f"final machine:{modified_machine_bsg}")
        
        return best_act["machine_bsg"],best_act["modifier_result"]
    
    def init_node(self,modified_machine_bsg,mcts_save_root,simulate_loop
                          ,designer_output = None
                          ,block_limitations=[]
                          ,required_feedback=None,task=None):
        machine_txt = modified_machine_bsg.replace(".bsg", ".txt")
        machine_result = machine_txt.replace("refined_machine", "output")
        threeDinfo = dp.get_3Dinfos_from_json(machine_txt)
        env_feedback=""
        env_feedback,scores = self.env_simulate(task=task,
                                        machine_bsg=modified_machine_bsg,
                                        skip_simulate=False,
                                        save=True,
                                        simulate_loop=simulate_loop,
                                        save_path=mcts_save_root,
                                        machine_txt=machine_txt,return_scores=True,
                                        designer_output = designer_output,
                                        threeDinfo = threeDinfo,
                                        block_limitations=block_limitations,
                                        required_feedback = required_feedback)

        if self.exp_type=="abl-self-critic":
            env_feedback=f"simulation score {scores}"
                
        mcts_node = {
            "node_id": -1,
            "parent_id": -1,
            "children_ids": [],
            "mcts_value":0, 
            "sim_value": scores, 
            "visit_count": 0,
            "machine_bsg": modified_machine_bsg,
            "machine_txt": machine_txt,
            "modifier_result":machine_result,
            "threeDinfo": threeDinfo,
            "designer_output": designer_output,
            "env_feedback": env_feedback,
            "simulate_loop": simulate_loop
        }
        return mcts_node
    
    
    def mcts_loop(self,modified_machine_bsg,modified_machine_output, save_path, designer_output, task, threeDinfo=None,
                mcts_search_times=3, 
                block_limitations=[]):

        def mcts(mcts_search_times,init_machine_bsg,mcts_save_root,init_machine_result=None,task=None):
            def calculate_ucb(children_ids, mct_tree,skip_nonvisited=False,get_sim_value = False):
                if get_sim_value:
                    best_sim_value = -float('inf')
                    best_child_node = None
                    for child_id in children_ids:
                        child_node = mct_tree[child_id]
                        if child_node["visit_count"] == 0:
                            if skip_nonvisited:
                                continue
                        if best_sim_value <child_node["sim_value"]:
                            best_sim_value = child_node["sim_value"]
                            best_child_node =child_node
                    return best_child_node
                
                best_ucb_value = -float('inf')
                best_child_node = None
                for child_id in children_ids:
                    child_node = mct_tree[child_id]
                    if child_node["visit_count"] == 0:
                        if not skip_nonvisited:
                            return child_node
                        else:
                            continue
                    else:
     
                        root_visit_count = mct_tree[0]["visit_count"]
                        ucb_value = (child_node["mcts_value"] / child_node["visit_count"]) + (2 * math.sqrt(math.log(root_visit_count) / child_node["visit_count"]))
                        if ucb_value > best_ucb_value:
                            best_ucb_value = ucb_value
                            best_child_node = child_node
                return best_child_node
            
            mct_tree = []
            init_required = None

            if init_machine_result is not None:
                init_machine_result_str = read_file(init_machine_result)
                init_required = self.get_required_list([init_machine_result_str])[0]

            root = self.init_node(modified_machine_bsg=init_machine_bsg,
                                mcts_save_root=mcts_save_root,
                                designer_output = designer_output,
                                block_limitations=block_limitations,
                                simulate_loop="0",
                                required_feedback=init_required,task=task)
            root["node_id"] = 0
            root["parent_id"] = -1
            mct_tree.append(root)
            for j in range(mcts_search_times):

                #selection
                should_exp= False
                can_rollout = False
                current_node = mct_tree[0]  
                while not can_rollout:
                    if len(current_node["children_ids"]) == 0:
                        if current_node["node_id"] == 0: 
                            should_exp = True
                            can_rollout = True
                        else:
                            if current_node["visit_count"] < 1:
                                can_rollout = True
                            else:
                                should_exp = True
                                can_rollout = True
                    else:
                        current_node = calculate_ucb(current_node["children_ids"], mct_tree) 
                #expand
                if should_exp:
                    _,modifier_result,modified_machine_json, machine_bsg = self.call_env_refiner(designer_output=designer_output,
                                                            machine_json=current_node["machine_txt"],
                                                            threeDinfo=threeDinfo,
                                                            env_feedback=current_node["env_feedback"],
                                                            save=True,
                                                            save_path=mcts_save_root,
                                                            block_limitations=block_limitations,
                                                            start_mcts_id=len(mct_tree)
                                                            )

                    if machine_bsg =="Error":
                        print("Warning: a MCTS leaf expand with no child, this may be not wrong, LLM may fail to create valid modify.")
                    else:
                        for i, child_machine_bsg in enumerate(machine_bsg):
                      
                            mct_tree_length = len(mct_tree)
                            child_node = self.init_node(modified_machine_bsg=child_machine_bsg,
                                                    mcts_save_root=mcts_save_root,
                                                    simulate_loop=f"{mct_tree_length}",
                                                    designer_output = designer_output,
                                                    block_limitations=block_limitations,task=task)
                                               
                            child_node["node_id"] = mct_tree_length
                            mct_tree.append(child_node)
                            child_node["parent_id"] = current_node["node_id"]
                            current_node["children_ids"].append(child_node["node_id"])
                        
                if len(current_node["children_ids"]) > 0:
                    child_node_id = current_node["children_ids"][0]
                    print(f"child_node_id: {child_node_id}, mct_tree length: {len(mct_tree)}")
                    child_node = mct_tree[child_node_id]
                    child_node["visit_count"] += 1
                    child_node["mcts_value"] += child_node["sim_value"]
                else:
                    child_node = current_node
                    child_node["visit_count"] += 1
                    child_node["mcts_value"] += child_node["sim_value"]
                # BackTrace
                current_node = mct_tree[current_node["parent_id"]]
                while current_node["parent_id"] != -1:
                    current_node["visit_count"] += 1
                    current_node["mcts_value"] += child_node["sim_value"]
                    current_node = mct_tree[current_node["parent_id"]]
                current_node["visit_count"] += 1
                current_node["mcts_value"] += child_node["sim_value"]
            

            best_act = calculate_ucb(list(range(0, len(mct_tree))), mct_tree,skip_nonvisited=True,get_sim_value=True)
            best_id = best_act["node_id"]
            with open(os.path.join(mcts_save_root, f"mcts_tree_best_{best_id}.json"), "w", encoding="utf-8") as f:
                json.dump(mct_tree, f, ensure_ascii=False, indent=4)
            return best_act["machine_bsg"],best_act["modifier_result"]
                    
        init_machine_result = modified_machine_output

        if os.path.exists(os.path.join(save_path, f"final_machine.bsg")):
            skip_simulate = True
            modified_machine_bsg = os.path.join(save_path, f"final_machine.bsg")
            init_machine_result = os.path.join(save_path, f"output.txt")
        else:
            skip_simulate = False
        if not skip_simulate:
            
            mcts_save_root = os.path.join(save_path, f"mcts_loop")
            os.makedirs(mcts_save_root, exist_ok=True)
            modified_machine_bsg,init_machine_result = mcts(
                mcts_search_times=mcts_search_times,
                init_machine_bsg=modified_machine_bsg,
                mcts_save_root=mcts_save_root,
                init_machine_result = init_machine_result,task=task)

            shutil.copy(modified_machine_bsg, os.path.join(save_path, f"final_machine_mcts.bsg"))
            base_path = os.path.splitext(modified_machine_bsg)[0]
            modified_machine_txt = base_path + ".txt"
            shutil.copy(modified_machine_txt, os.path.join(save_path, f"final_machine_mcts.txt"))
            shutil.copy(init_machine_result, os.path.join(save_path, f"output_mcts.txt"))
        print(f"final machine: {modified_machine_bsg}")

    def env_modify_loop(self, modified_machine_bsg_paths,designer_output,
                        block_limitations=[],
                        mcts_search_times=5,
                        search_strategy="random"):
        
        if not isinstance(designer_output, list):
            designer_output = [designer_output]*len(modified_machine_bsg_paths)
        
        modified_machine_bsgs=[]
        modified_machine_output=[]
        threeDinfos=[]
        for bsg_path in modified_machine_bsg_paths:
            for p_file in  os.listdir(bsg_path):
                if p_file in MACHINE_TXT_FILENAMES:
                    
                    try:
                        machine_tree = read_txt(os.path.join(bsg_path,p_file))
                        threeDinfos.append(dp.get_3Dinfos_from_json(machine_tree))
                        modified_machine_bsgs.append(os.path.join(bsg_path,p_file.replace(".txt",".bsg")))
                        modified_machine_output.append(os.path.join(bsg_path,"output.txt"))
                        break
                    except:
                        pass
        
          
        save_paths=[]      
        for i,inspector_path in enumerate(modified_machine_bsg_paths):
            save_paths.append(inspector_path.replace(self.agentic_paths["refiner"],self.agentic_paths["env_loop"]))
        
        self._initialize_env_manager()
        
        if search_strategy == "random":
            threads = []
            def thread_target(index,max_loop,block_limitations):
                self.single_loop(modified_machine_bsg=modified_machine_bsgs[index],
                                 modified_machine_output=modified_machine_output[index],
                            save_path=save_paths[index],
                            designer_output=designer_output[index],
                            threeDinfo=threeDinfos[index],
                            max_loop=max_loop,
                            block_limitations=block_limitations,
                            task=self.tasks[0])
            
            for i in range(len(modified_machine_bsgs)):
                print(f"{i}-th machine env loop")
                thread = threading.Thread(target=thread_target, args=(i,mcts_search_times,block_limitations))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()  
            print("all env-loop done")
        elif search_strategy == "bon":
            threads = []
            def thread_target(index,max_loop,block_limitations):
                self.best_of_n_env_loop(modified_machine_bsg=modified_machine_bsgs[index],
                                        modified_machine_output=modified_machine_output[index],
                            save_path=save_paths[index],
                            designer_output=designer_output[index],
                            threeDinfo=threeDinfos[index],
                            max_loop=max_loop,
                            block_limitations=block_limitations,
                            task=self.tasks[0])
            
            for i in range(len(modified_machine_bsgs)):
                print(f"{i}-th machine env loop")
                thread = threading.Thread(target=thread_target, args=(i,mcts_search_times,block_limitations))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()  

            print("all env-loop done")
        elif search_strategy == "MCTS":

            threads = []
            def thread_target(index,block_limitations,mcts_search_times):
                self.mcts_loop(modified_machine_bsg=modified_machine_bsgs[index],
                               modified_machine_output=modified_machine_output[index],
                            save_path=save_paths[index],
                            designer_output=designer_output[index],
                            threeDinfo=threeDinfos[index],
                            mcts_search_times = mcts_search_times,
                            block_limitations=block_limitations,
                            task=self.tasks[0])
            
            for i in range(len(modified_machine_bsgs)):
                print(f"{i}-th machine env loop")
                thread = threading.Thread(target=thread_target, args=(i,block_limitations,mcts_search_times))
                threads.append(thread)
                thread.start()
            
            for thread in threads:
                thread.join()

            print("all env-loop done")
            
    def env_simulate(self,task,machine_bsg,simulate_loop=0,skip_simulate=False,save=False,save_path=None,machine_txt=None,
                     use_querier=True,
                     required_feedback = None,
                     designer_output = None,
                     block_limitations=[],
                     threeDinfo = None,
                     return_scores=False,return_raw_result=False):        
        #instructions need_infos win_condition limit_blocks human_input
        self._initialize_env_manager()
        
        simulate_menu = LEVELMENUS[task]
        

        if skip_simulate:
            raw_simulate_results_path = os.path.join(os.path.dirname(machine_bsg),f"env_simulate_{simulate_loop}.txt")
            raw_simulate_results = ast.literal_eval(read_txt(raw_simulate_results_path))
        else:
            
            
            simulate_env = self.env_manager.get_available_environment(task)
            

            instruct_lists = simulate_menu["instructions"]
            raw_simulate_results = besiege_level_menus(simulate_env,machine_bsg,instruct_lists)

            self.env_manager.release_environment(simulate_env)

        if save:
            if save_path==None:
                raw_simulate_results_path = os.path.join(os.path.dirname(machine_bsg),f"env_simulate_{simulate_loop}.txt")
            else:
                raw_simulate_results_path = os.path.join(save_path,f"env_simulate_{simulate_loop}.txt")
            write_file(raw_simulate_results_path,str(raw_simulate_results))


        bsg_machine = dp.process_bsg(machine_bsg)
        # print(bsg_machine)

        #print(raw_simulate_results)
        sim_time,game_state_dict,is_win = self.process_env_outputs(raw_simulate_results,bsg_machine)
        # win_condition = simulate_menu["win_condition"]

        if machine_txt is not None:
            machine_json = extract_json_from_string(machine_txt)
        else:
            machine_json=None
        
        return_result=[]

        if self.exp_type=="abl-env_querier" or self.exp_type=="abl-self-critic":
            ###############################ablation
            use_querier = False

        try:
            env_feedback = dp.get_env_feedback(
                sim_time, game_state_dict, simulate_menu, bsg_machine,
                machine_json=machine_json, return_scores=return_scores,
                required_feedback = required_feedback,
                use_querier=use_querier,
                designer_output = designer_output,
                threeDinfo = threeDinfo,
                block_limitations=block_limitations,
                save_path=save_path,
                simulate_loop = simulate_loop,
                agentic_pipeline = self,
                is_win=is_win
            )
            if return_scores:
                env_feedback, scores = env_feedback
                return_result.append(env_feedback)
                return_result.append(scores)
            else:
                return_result.append(env_feedback)
            if return_raw_result:
                return_result.append(sim_time)
                return_result.append(game_state_dict)
                return_result.append(bsg_machine)
            
            return tuple(return_result)
            
        except Exception as e:
            print(f"Error processing environment outputs: {e}")
            print(traceback.format_exc())

            env_feedback = "env simulate error"
            if return_scores:
                return_result.append(env_feedback)
                return_result.append(0)
            else:
                return_result.append(env_feedback)
            
            if return_raw_result:
                return_result.append(sim_time)
                return_result.append(game_state_dict)
                return_result.append(bsg_machine)
            
            return tuple(return_result)
    
    
    def _process_envfeedback_for_refiner(self,designer_output,threeDinfo,block_limitations,machine_json,env_feedback):
        task_definition = extract_json_from_string(designer_output["designer_output"])["definition"]
        env_feedback = env_feedback
        threedinfo = threeDinfo
        block_limitations =list({int(block['tid']) for block in threeDinfo 
                        if isinstance(block, dict) and 'tid' in block})
        block_limitations = list(set(block_limitations+block_limitations))
        json_info = machine_json

        fail_history = "None"
        
        kargs = {
            "task_definition":task_definition,
            "jsoninfo":json_info,
            "threedinfo":threeDinfo,
            "env_feedback":env_feedback,
            "block_limitations":block_limitations,
            "fail_history":fail_history
        }
        return kargs
    

    def call_env_refiner(self,designer_output,machine_json,threeDinfo,
                          env_feedback,save = True,save_path=None,
                          block_limitations=[],
                          start_mcts_id=0,parallel_num=4):

        if os.path.exists(machine_json):
            machine_json = extract_json_from_string(machine_json)
        
        kargs = self._process_envfeedback_for_refiner(designer_output=designer_output,threeDinfo=threeDinfo,block_limitations=block_limitations,machine_json=machine_json,env_feedback=env_feedback)
        
        total_input,refiner_output,refiner_machine = self.run_refiner(kargs = kargs,need_all_success=True,env_refine=True,parallel_num=parallel_num)
        if save:
            modified_machine_save_paths=[]
            
            final_total_input_list=[]
            final_result_list=[]
            final_modified_machine_json_list=[]
            success_num=0
            
            for i,success in enumerate(refiner_machine):
                if not success: continue
                success_num+=1
                final_total_input_list.append(total_input[i])
                final_result_list.append(refiner_output[i])
                final_modified_machine_json_list.append(refiner_machine[i])
            print(f"final success refine num {success_num}")
            total_input_list = final_total_input_list
            result_list=final_result_list
            modified_machine_json_list=final_modified_machine_json_list
            
            for j in range(len(modified_machine_json_list)):
                leaf_id = start_mcts_id+j
                sim_loop = f"_{leaf_id}"
                modifyer_input_save_path = os.path.join(save_path,f"env_input{sim_loop}.txt")
                modify_result_save_path = os.path.join(save_path,f"env_output{sim_loop}.txt")
                modified_machine_save_path = os.path.join(save_path,f"env_refined_machine{sim_loop}.txt")
                modified_machine_save_paths.append(modified_machine_save_path.replace(".txt",".bsg"))
                txt_file_paths = [modifyer_input_save_path,modify_result_save_path,modified_machine_save_path]
                modifyer_input = total_input_list[j]
                modified_machine_json_str = json.dumps(modified_machine_json_list[j])
                modified_machine_json_str = f"```json{modified_machine_json_str}```"
                txt_contents = [modifyer_input,result_list[j],modified_machine_json_str]
                for k,txt_file_path in enumerate(txt_file_paths):
                    write_file(txt_file_path,str(txt_contents[k]))
            self.save_machine()
            return total_input_list,result_list,modified_machine_json_list,modified_machine_save_paths
            
        return total_input,refiner_output,refiner_machine,"No Save"
    

    def call_inspector(self, valid_chain, save=True, quiz_sample_each_machine=2,max_retry_time=5, skip_inspector=False, block_limitations=[]):
        if skip_inspector:
            save = False

        quizes, quiz_inputs, threeDinfos, machine_jsons = [], [], [], []
        save_paths, valid_machine_paths = [], []

        valid_chain=self.agentic_paths["structure_chain"]
        
        for chain in valid_chain:
            designer_id = chain.split("-")[-1]
            valid_machine_root = os.path.join(self.agentic_paths["designer"], f"designer_{designer_id}", chain)
            valid_machine_path = os.path.join(valid_machine_root, "output.txt")
            
            inspector_save_path = os.path.join(self.agentic_paths["inspector"],chain)
            
            save_paths.append(inspector_save_path)
            valid_machine_paths.append(valid_machine_path)
            
            builder_content = read_txt(valid_machine_path)
            if self.exp_type=="abl-3dinfo":
                threeDinfos.append([])
            else:
                threeDinfos.append(dp.get_3Dinfos_from_json(builder_content))
            machine_jsons.append(extract_json_from_string(valid_machine_path))
            
            if skip_inspector:
                try:
                    quiz_path = os.path.join(inspector_save_path, "output.txt")
                    quiz_input_path = os.path.join(inspector_save_path, "input.txt")
                    quizes.append(read_txt(quiz_path))
                    quiz_inputs.append(read_txt(quiz_input_path))
                except:
                    pass

        if not skip_inspector:
            def process_quiz_sample(path):
                for attempt in range(max_retry_time):
                    try:
                        total_input, result, kargs = self.ask_inspector(path, block_limitations=block_limitations)
                        if quizzer_format_checking(result):
                            return result, total_input, kargs
                    except Exception as e:
                        print(e)
                        print("Stack trace:")
                        traceback.print_exc()
                        time.sleep(1)
                return None, None, None

            def process_machine_path(path):
                result, total_input, kargs = process_quiz_sample(path)
                return result, total_input

            # Main processing
            with ThreadPoolExecutor() as executor:
                results = executor.map(process_machine_path, valid_machine_paths)
                quizes = []
                quiz_inputs = []
                
                for result, total_input in results:
                    quizes.append(result)
                    quiz_inputs.append(total_input)

        if save:
            for i, path in enumerate(save_paths):
                os.makedirs(path,exist_ok=True)
                
                quiz_input_path = os.path.join(path, "input.txt")
                quiz_path = os.path.join(path, "output.txt")
                if quizes[i]!=None:
                    write_file(quiz_input_path, quiz_inputs[i])
                    write_file(quiz_path, quizes[i])

        return quizes, quiz_inputs, save_paths, threeDinfos, machine_jsons
    

    def _process_inspector_for_refiner(self,quiz,threeDinfo,machine_json,task_definition,block_limitations):
        try:
            conclusion = match_case(quiz, "<Summary of Design Defects>", "</Summary of Design Defects>")[0]
        except:
            conclusion=""
            
        if self.exp_type=="abl-quizzer":
            conclusion = "" 
        related_blocks = list(dict.fromkeys(
            int(float(item.replace("[", "").replace("]", "").strip()))
            for block in match_case(conclusion, "[", "]")
            for item in block.split(',') if item.strip().isdigit()
        ))
        if self.exp_type=="abl-auto-3dinfo":
            try:
                if_3d = match_case(quiz, "<Give 3D Info>", "</Give 3D Info>")[0]
                if "False" in if_3d:
                    threeDinfo=[]
                    block_limitations = list(set(block_limitations))
                    focus_blocks = []
                else:
                    block_limitations = list({
                        int(block['tid']) for block in threeDinfo 
                        if isinstance(block, dict) and 'tid' in block
                    } | set(block_limitations))
                    focus_blocks = [block for block in threeDinfo if block['id'] in related_blocks]
            except:
                block_limitations = list(set(block_limitations))
                focus_blocks = []
                
        if self.exp_type=="abl-3dinfo":
            block_limitations = list(set(block_limitations))
            focus_blocks = []
        else:
            block_limitations = list({
                int(block['tid']) for block in threeDinfo 
                if isinstance(block, dict) and 'tid' in block
            } | set(block_limitations))
            focus_blocks = [block for block in threeDinfo if block['id'] in related_blocks]
        focus_blocks = [block for block in threeDinfo if block['id'] in related_blocks]
        kargs = {
            "task_definition": task_definition,
            "jsoninfo": machine_json,
            "threedinfo": threeDinfo,
            "focus_info": focus_blocks,
            "focus_report": conclusion,
            "block_limitations": block_limitations,
            "fail_history": "None"
        }
        
        return kargs

    def evaulate_refiner_results(self,sample_results, fail_history_record,successed_sample,max_retry_time,processed_sample_results=None):
        if not processed_sample_results:
            refiner_results = [None]*len(sample_results)
        else:
            refiner_results = processed_sample_results
        has_success_results=False
        all_success=True
        for i,sample in enumerate(sample_results):
            if not sample or successed_sample[i]:
                continue
            
            step_input, result, kargs = sample
            try:
                modify_step = match_case(result, "<Modification Steps>", "</Modification Steps>")[0]
            except:
                continue
            modify_steps = [line.strip() for line in modify_step.split('\n') if line.strip()]
            
            modify_return = dp.modify_json(kargs["jsoninfo"], modify_steps)
            if not isinstance(modify_return, dict):
                has_success_results=True
                successed_sample[i]=True
                refiner_results[i]={
                    "refine_result": modify_return[0], "step_input": step_input,
                    "result": result, "kargs": kargs, "modify_step": modify_steps, "fail_history": None,"success_rate":1,"max_retry_time":max_retry_time}
            else:
                all_success=False
                fail_history, success_rate = dp.generate_modify_history(modify_steps, modify_return)
                operation = modify_return["operation"]
                fail_reason = modify_return["reason"]
                fail_step = str(modify_steps[:modify_steps.index(operation)]) if operation in modify_steps else ""
                fail_history_record.setdefault(fail_step, {})[operation] = fail_reason

                refiner_results[i]={
                    "refine_result": None, "step_input": step_input,
                    "result": result, "kargs": kargs, "modify_step": modify_steps, "fail_history": fail_history,"success_rate":success_rate,"max_retry_time":max_retry_time}

        return refiner_results,fail_history_record,has_success_results,all_success,successed_sample

    def run_refiner(self,kargs,parallel_num=4,retry_limit=5,need_all_success=False,env_refine=False):
        fail_history_record = {}
        total_input = [""]*parallel_num
        
        successed_sample = [False]*parallel_num
        
        kargs_list = []
        for _ in range(parallel_num):
            kargs_list.append(copy.deepcopy(kargs))
            
        max_retry_time = retry_limit
        
        old_processed_sample_results=None
        processed_sample_results=None
        while max_retry_time > 0:
            max_retry_time -= 1
            old_processed_sample_results = processed_sample_results
            processed_sample_results=[None]*parallel_num
            sample_results = [None]*parallel_num
            
            while all(x is None for x in sample_results):
                futures=[]
                with ThreadPoolExecutor(max_workers=4) as executor:
                    for p in range(parallel_num):
                        if successed_sample[p]:
                            futures.append(None)
                        else:
                            futures.append(executor.submit(self.ask_refiner, kargs_list[p],env_refine))
                    for i, fut in enumerate(futures): 
                        try:
                            if fut==None:
                                sample_results[i]=None
                            else:
                                sample_results[i] = fut.result() 
                        except Exception as e:
                            print(f"error running refiner: {str(e)}")
                            print(traceback.format_exc())

            #overwrite successed results
            for i,successed in enumerate(successed_sample):
                if successed:
                    processed_sample_results[i] = old_processed_sample_results[i]
            
            # here we return fail_history_record is for update it
            processed_sample_results,fail_history_record,has_success_results,all_success,successed_sample = self.evaulate_refiner_results(sample_results, fail_history_record,successed_sample,max_retry_time,processed_sample_results)
            
            if all(x is None for x in sample_results):
                print("all refine results are illegal format retry...")
                continue

            for i,refine_result in enumerate(processed_sample_results):
                if not refine_result:
                    continue
                
                if refine_result["kargs"]["fail_history"] == "None":
                    #here we record the system and user prompt
                    total_input[i] += refine_result["step_input"]
                if not has_success_results:
                    if successed_sample[i]:
                        continue
                    
                    kargs = refine_result["kargs"]
                    kargs["fail_history"] = refine_result["fail_history"].replace("fail_history:", "")
                    #update fail history
                    modify_step_list = refine_result["modify_step"]
                    for line_history in refine_result["fail_history"].splitlines():
                        if "Error:" in line_history:
                            operation = line_history.split("Error:")[0].strip()
                            break
                    for j, item in enumerate(modify_step_list):
                        if item == operation:
                            fail_step_history = str(modify_step_list[:j])
                            if fail_step_history in fail_history_record:
                                kargs["fail_history"] += f"\nFail Steps, Do Not Use:\n{list(fail_history_record[fail_step_history].keys())}"
                            break

                    if self.exp_type=="abl-modifier-history":
                        ###############################ablation
                        kargs['fail_history']="" 
                kargs_list[i]["fail_history"] = refine_result["kargs"]["fail_history"]
            
            if has_success_results:
                if need_all_success and all_success:
                    print("!!! refine success !!!")
                    max_retry_time=-1
                if not need_all_success:
                    print("!!! refine success !!!")
                    max_retry_time=-1
            else:
                print("refine not success, retrying...")
                pass
        
        refiner_output=[None]*parallel_num
        refiner_machine=[None]*parallel_num
        
        for i,refine_result in enumerate(processed_sample_results):
            total_input[i] += "-fail_history_steps_exp-"
            total_input[i] += str(max_retry_time)
            total_input[i] += "-----"
            total_input[i] += str(refine_result.get("modify_step", ""))
            refiner_output[i]=refine_result["result"]
            refiner_machine[i]=refine_result["refine_result"]
        
        return total_input,refiner_output,refiner_machine


    def call_refiner(self, designer_output, quizes, save_paths, threeDinfos, machine_jsons, save=True,
                 skip_refiner=False, block_limitations=[],retry_limit=5):

        def search_valid_results(save_paths):
            results = []
            for path in save_paths:
                input_file = os.path.join(path, "input.txt")
                output_file = os.path.join(path,"output.txt")
                machine_file = os.path.join(path,"refined_machine.txt")
                if all(os.path.exists(f) for f in [input_file, output_file, machine_file]):
                    results.append((
                        read_txt(input_file),
                        read_txt(output_file),
                        extract_json_from_string(machine_file),
                        path
                    ))
                else:
                    pass
            
            return tuple(zip(*results)) if results else ([], [], [],[])
        
        
        for i,inspector_path in enumerate(save_paths):
            save_paths[i] = inspector_path.replace(self.agentic_paths["inspector"],self.agentic_paths["refiner"])
        
        if skip_refiner:
            return search_valid_results(save_paths)

        task_definition = extract_json_from_string(designer_output["designer_output"])["definition"]
        total_inputs= [None]*len(quizes)
        modify_results= [None]*len(quizes)
        modified_machine_jsons = [None]*len(quizes)
        
        if self.exp_type!="abl-modifier-history" and self.exp_type!="abl-quizzer":
            existing_indices = [i for i, path in enumerate(save_paths) if all(\
                os.path.exists(os.path.join(path, f)) for f in [\
                    "modifyer_input.txt", "modifyer.txt", "modified_machine.txt"])]
        else:
            existing_indices=[]
        save_paths = [path for i, path in enumerate(save_paths) if i not in existing_indices]
        quizes = [quiz for i, quiz in enumerate(quizes) if i not in existing_indices]

        for i, quiz in enumerate(quizes):
            print(f"processing machine {i}")
            threeDinfo = threeDinfos[i]
            machine_json= machine_jsons[i]
            kargs = self._process_inspector_for_refiner(quiz,threeDinfo,machine_json,task_definition,block_limitations)
            
            total_input,refiner_output,refiner_machine = self.run_refiner(kargs = kargs,need_all_success=False)
            
            for j,refined_machine in enumerate(refiner_machine):
                if not refined_machine: continue
                modify_results[i]=refiner_output[j]
                modified_machine_jsons[i]=refiner_machine[j]
                total_inputs[i]=total_input[j]
                break
            

            if save and total_inputs[i]!=None:
                try:
                    save_path = save_paths[i]
                    os.makedirs(save_path,exist_ok=True)
                    contents = [
                        total_inputs[i],
                        modify_results[i],
                        f"```json{json.dumps(modified_machine_jsons[i])}```"
                    ]
                    for j, name in enumerate(["input.txt", "output.txt", "refined_machine.txt"]):
                        write_file(os.path.join(save_path, name), str(contents[j]))
                    self.save_machine()
                except Exception as e:
                    print(e)
                    pass
        
        return search_valid_results(save_paths)

    def ask_refiner(self,kargs,
                     env_refine=False):

        if env_refine:
            refiner_prompt_type = "env_short"
        else:
            refiner_prompt_type = "short"
        user_input = format_agentic_input(kargs,"refiner",refiner_prompt_type)
            
        modifyer_agent,custome_prompt = self.prepare_agent_auto(temperature=0.7,agent="refiner",
                                                refiner_prompt_type=refiner_prompt_type,need_prompt=True,
                                                block_limitations=kargs["block_limitations"])
        modifyer_agent = modifyer_agent[0]

        result = self.chat_once(modifyer_agent,input=user_input)["content"]
        total_input = f"system:{custome_prompt},user:{user_input}"
        return total_input,result,kargs
    
    def ask_env_querier(self,kargs):

        env_querier_prompt_type = "short"

        user_input = format_agentic_input(kargs,"env_querier",env_querier_prompt_type)
        
        env_querier_agent,custome_prompt = self.prepare_agent_auto(temperature=0.7,agent="env_querier",
                                                env_querier_prompt_type=env_querier_prompt_type,need_prompt=True,
                                                block_limitations=kargs["block_limitations"])
        env_querier_agent = env_querier_agent[0]

        result = self.chat_once(env_querier_agent,input=user_input)["content"]
        total_input = f"system:{custome_prompt},user:{user_input}"
        return total_input,result,kargs

    def process_kargs_for_inspector(self, designer_output_path,block_limitations=[]):
        designer_dir = Path(designer_output_path)
        # print(f"designer_dir {designer_dir}")
        
        try:
            meta_designer_dir = Path(self.agentic_paths["meta-designer"])
            meta_designer_output = extract_json_from_string(meta_designer_dir/ "output.txt")
            build_order = meta_designer_output.get("build_order", [])
            task_definition = meta_designer_output.get("definition", "")
        except Exception as e:
            # print(e)
            if not self.custome_designer_output:
                print('no meta-designer info')
            build_order = []
            task_definition = ""
        
        build_structure_dict = {structure: [] for structure in build_order}
        
        # print(designer_dir.parent.name)
        structure_chain = designer_dir.parent.name.split("_")[1]
        current_chain = ""
        
        for i, part in enumerate(structure_chain.split("-")):
            current_chain += part
            builder_folder = f"designer_{part}"
            builder_subfolder = f"structurechain_{current_chain}"
            
            substructure_path = Path(self.agentic_paths["designer"]) / builder_folder / builder_subfolder / "output.txt"
            current_chain += "-"
            if not os.path.exists(substructure_path):
                continue
            substructure_content = extract_json_from_string(substructure_path)
            for block_json in substructure_content:
                build_structure_dict[build_order[i]].append(block_json["order_id"])
        
        # print(build_structure_dict)
        
        result = []
        for sublist in build_structure_dict.values():
            if result:
                sublist = [item for item in sublist if item not in 
                        [item for sub in result for item in sub]]
            result.append(sublist)
        
        builder_content = extract_json_from_string(designer_dir)
        block_limitations = list({int(block['id']) for block in builder_content 
                        if isinstance(block, dict) and 'id' in block})
        block_limitations = list(set(block_limitations+block_limitations))
        
        builder_content_string = read_txt(designer_dir)

        return {
            "task_definition": task_definition,
            "json_file": builder_content,
            "threedinfo": dp.get_3Dinfos_from_json(builder_content_string),
            "structure_info": dict(zip(build_order, result)),
            "block_limitations": block_limitations
        }
    
    def ask_inspector(self,builder_file_path,block_limitations=[]):
        kargs = self.process_kargs_for_inspector(builder_file_path,block_limitations=block_limitations)
        # print(kargs)
        
        inspector_prompt_type = "short"
            
        if self.exp_type=="abl-auto-3dinfo":
            inspector_prompt_type = "short-auto-3dinfo"


        inspector_agent,inspector_prompt = self.prepare_agent_auto(temperature=0.7,agent="inspector"
                                                            ,block_limitations=kargs["block_limitations"],
                                                            need_prompt=True,
                                                            inspector_prompt_type=inspector_prompt_type)
        inspector_agent =inspector_agent[0]
        
        user_input = format_agentic_input(kargs,"inspector",inspector_prompt_type)
        
        result = self.chat_once(inspector_agent,input=user_input)["content"]
        total_input = f"system:{inspector_prompt},user:{user_input}"

        return total_input,result,kargs

    @staticmethod
    def process_env_outputs(env_outputs,bsg_machine):
        def calculate_angular_velocity(new_rotation, old_rotation, time_interval):
            delta_rotation = R.from_quat(new_rotation) * R.from_quat(old_rotation).inv()
            return np.round(delta_rotation.as_euler('xyz', degrees=True) / time_interval, 2)

        def calculate_velocity(new_position, old_position, time_interval):
            velocity_vector = [round((new - old) / time_interval, 2) for new, old in zip(new_position, old_position)]
            v = round(math.sqrt(sum(v ** 2 for v in velocity_vector)), 2)
            return velocity_vector, v

        is_win=False
        for output in env_outputs:
            if "Win" in output:
                is_win=True

        cleaned_outputs = []
        for output in env_outputs:
            if "GameStat:" in output:
                cleaned_outputs.append(output.split("GameStat:")[1].strip())

        try:
            init_game_state = cleaned_outputs[0]
        except:
            init_game_state=[]
        init_game_state = [part for part in init_game_state.split("|") if part]
        block_name_list = []
        for obj_info in init_game_state:
            block_name_list.append(obj_info.split(":")[0])

        game_state_dict = []
        for i,game_state in enumerate(cleaned_outputs):
            for j,single_obj_info in enumerate([part for part in game_state.split("|") if part]):
                obj_name,traj = single_obj_info.split(":")
                traj = traj.split(",")
                pos = np.array([float(traj[0]),float(traj[1]),float(traj[2])])
                quat = np.array([float(traj[3]),float(traj[4]),float(traj[5]),float(traj[6])])
                
                stress = traj[7]
                
                if i==0:
                    obj_info = {}
                    obj_info["name"]=obj_name
                    obj_info["pos"]= [pos]
                    obj_info["quat"] = [quat]
                    obj_info["alive"] = 0
                    obj_info["vel"]=[]
                    obj_info["rot_vel"]=[]
                    obj_info["in_machine"] = 0
                    obj_info["orderid"]=j
                    obj_info["stress"] = [stress]
                    
                    if len(game_state_dict)==0:
                        start_block_pos=pos
                    else:
                        start_block_pos = game_state_dict[0]["pos"][i]

                    if "GOAL_" in obj_name:
                        init_startblock_dis = np.abs(np.array(pos) - np.array(start_block_pos))
                        init_startblock_dis = np.round(init_startblock_dis, 2)
                        init_startblock_dis = round(np.linalg.norm(init_startblock_dis), 2)
                        obj_info["init_startblock_dis"] = init_startblock_dis
                    else:
                        if j>=len(bsg_machine):
                            print(j)
                        obj_info["init_startblock_dis"] = bsg_machine[j]["startblock_dis"]

                    startblock_dis = np.abs(np.array(pos) - np.array(start_block_pos))
                    startblock_dis = np.round(startblock_dis, 2)
                    startblock_dis = round(np.linalg.norm(startblock_dis), 2)
                    obj_info["startblock_dis"]=[startblock_dis]
                    game_state_dict.append(obj_info)
                else:
                    if game_state_dict[j]["name"]==obj_name:
                        obj_info = game_state_dict[j]
                    else:
                        for offset in range(len(game_state_dict)-j):
                            if game_state_dict[j+offset]["name"]==obj_name:
                                obj_info = game_state_dict[j+offset]
                                break
                    
                    if "_broken" in obj_name:
                        continue

                    old_pos = obj_info["pos"][-1]
                    vel_dir,vel_digital = calculate_velocity(pos,old_pos,LIFECYCLE)
                    old_quat = obj_info["quat"][-1]
                    rot_vel = calculate_angular_velocity(quat,old_quat,LIFECYCLE)
                    obj_info["pos"].append(pos)
                    obj_info["quat"].append(quat)
                    obj_info["vel"].append(vel_dir)
                    obj_info["rot_vel"].append(rot_vel)
                    obj_info["alive"]+=LIFECYCLE
                    obj_info["stress"].append(stress)

                    start_block_pos = game_state_dict[0]["pos"][i]
                    
                    startblock_dis = np.abs(np.array(pos) - np.array(start_block_pos))
                    startblock_dis = np.round(startblock_dis, 2)
                    startblock_dis = round(np.linalg.norm(startblock_dis), 2)

                    startblock_dis_list = obj_info["startblock_dis"]
                    startblock_dis_list.append(startblock_dis)
                    if abs(startblock_dis_list[-1]-obj_info["init_startblock_dis"])<0.5:
                        obj_info["in_machine"] += LIFECYCLE
                    
        sim_time  = (len(cleaned_outputs)-1)*LIFECYCLE
        return sim_time,game_state_dict,is_win
    
    ################

    def process_response_auto(self, response, last_agent, old_kargs,block_limitations=None,task=None):
        try:
            if last_agent =="single":
                check_result = dp.valid_check(response)
                if check_result is True:
                    random_name = generate_random_string()
                    tmp_path = os.path.join(old_kargs["tmp_path"], f"{random_name}.bsg")
                    dp.json_to_xml(response,tmp_path)
                    env_feedback = self.env_simulate(task=self.tasks[0],
                                    machine_bsg=tmp_path,
                                    use_querier=False,
                                    return_scores=False
                                    )
                    if os.path.exists(tmp_path):
                        os.remove(tmp_path)
                    return {
                        "env_feedback": env_feedback,
                    }
                else:
                    return False
        except Exception:
            return False

if __name__ == "__main__":   
    pass  