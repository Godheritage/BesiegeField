from peft import PeftModel
from peft import get_peft_model,OFTConfig,PeftConfig, TaskType
import torch
import os
from safetensors.torch import load_file
from transformers import AutoModelForCausalLM, AutoTokenizer,BitsAndBytesConfig
from deepspeed.utils.zero_to_fp32 import get_fp32_state_dict_from_zero_checkpoint


base_model_path   = "base_model_path"
ckpt_name = ["ckpt_num"]
ckpt_root    = "ckpt_root" 
export_model_path = "export_model_path"

def merge_deepspeed_ckpt(ckpt_name="ckpt_num",
                        oft_ckpt_path="ckpt_root",
                        export_model_path="export_model_path"):
    oft_ckpt_path    += ckpt_name  
    export_model_path += ckpt_name


    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )
    oft_config = OFTConfig(
        r=0,
        oft_block_size=64,
        use_cayley_neumann=True,
        target_modules="all-linear",
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )
    model = get_peft_model(model, oft_config)
    state_dict = get_fp32_state_dict_from_zero_checkpoint(oft_ckpt_path)

    model.load_state_dict(state_dict, strict=False)

    model = model.merge_and_unload()  

    model.save_pretrained(export_model_path, safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(base_model_path)
    tokenizer.save_pretrained(export_model_path)

    
for ckpt in ckpt_name:
    merge_deepspeed_ckpt(ckpt,ckpt_root,export_model_path)
