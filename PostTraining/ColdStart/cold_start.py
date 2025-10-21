import json, os, torch, argparse
from datasets import load_dataset,load_from_disk,DatasetDict
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import OFTConfig, TaskType
from trl import SFTTrainer,SFTConfig
from transformers.trainer_utils import get_last_checkpoint
import shutil
import subprocess
from pathlib import Path

LOAD_HUMAN_DATA=False

def load_dataset_from_hf(dataset_name, train_ratio=0.9, seed=42):
    dataset = load_dataset(dataset_name)
    if "train" in dataset and "dev" in dataset:
        return dataset
    
    split_name = list(dataset.keys())[0]
    dataset = dataset[split_name]
    
    print(f"dataset total: {len(dataset)}")
    
    total_size = len(dataset)
    train_size = int(total_size * train_ratio)
    
    dataset = dataset.shuffle(seed=seed)
    train_dataset = dataset.select(range(train_size))
    dev_dataset = dataset.select(range(train_size, total_size))
    
    print(f"train: {len(train_dataset)}")
    print(f"dev: {len(dev_dataset)}")
    raw = DatasetDict({
        "train": train_dataset,
        "dev": dev_dataset
    })
    
    return raw

def load_human_dataset(
    repo_id="Godheritage/BesiegeField_humandataset_coldstart",
    output_dir="dataset/human_dataset",
    split=True,
    train_ratio=0.9,
    seed=42,
    force_download=False
):
    """
    Load human dataset with auto-caching and optional train/dev split
    
    Args:
        repo_id: HF repo ID
        output_dir: Cache directory
        split: Return DatasetDict if True, else Dataset
        train_ratio: Train/dev ratio
        seed: Random seed
        force_download: Force re-download
    
    Returns:
        DatasetDict or Dataset
    """
    parquet_file = Path(output_dir) / "data.parquet"
    
    # Load from cache if exists
    if parquet_file.exists() and not force_download:
        print(f"✓ Loading from cache: {parquet_file}")
        dataset = Dataset.from_parquet(str(parquet_file))
        print(f"✓ Loaded {len(dataset)} items")
    else:
        # Download and convert
        print(f"Downloading from {repo_id}...")
        temp_dir = "temp_human_dataset"
        
        try:
            # Git clone
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)
            subprocess.run(
                ["git", "clone", f"https://huggingface.co/datasets/{repo_id}", temp_dir],
                capture_output=True, text=True, check=True
            )
            print(f"✓ Cloned to {temp_dir}")
            
            # Process data
            dataset_path = Path(temp_dir)
            with open(dataset_path / "human_dataset_prompt.txt", 'r', encoding='utf-8') as f:
                system_prompt = f.read().strip()
            
            data_list = []
            trainable_dir = dataset_path / "trainable_dataset"
            subdirs = [d for d in trainable_dir.iterdir() if d.is_dir()]
            
            print(f"Processing {len(subdirs)} items...")
            for idx, subdir in enumerate(subdirs, 1):
                try:
                    with open(subdir / "construction_target.txt", 'r', encoding='utf-8') as f:
                        user_content = f.read().strip()
                    json_file = list(subdir.glob("*.json"))[0]
                    with open(json_file, 'r', encoding='utf-8') as f:
                        assistant_content = json.dumps(json.load(f), ensure_ascii=False)
                    
                    data_list.append({'messages': [
                        {'content': system_prompt, 'role': 'system'},
                        {'content': user_content, 'role': 'user'},
                        {'content': assistant_content, 'role': 'assistant'}
                    ]})
                    
                    if idx % 100 == 0:
                        print(f"  {idx}/{len(subdirs)}")
                except Exception as e:
                    print(f"  Skip {subdir.name}: {e}")
            
            print(f"✓ Processed {len(data_list)} items")
            
            # Save to parquet
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            dataset = Dataset.from_dict({"messages": [item["messages"] for item in data_list]})
            dataset.to_parquet(parquet_file)
            print(f"✓ Saved to {parquet_file}")
            
            # Cleanup
            shutil.rmtree(temp_dir)
            
        except Exception as e:
            print(f"✗ Error: {e}")
            print("Tip: Run 'git config --global credential.helper store' and 'huggingface-cli login'")
            raise
    
    # Split if requested
    if split:
        total = len(dataset)
        train_size = int(total * train_ratio)
        dataset = dataset.shuffle(seed=seed)
        
        result = DatasetDict({
            "train": dataset.select(range(train_size)),
            "dev": dataset.select(range(train_size, total))
        })
        print(f"✓ Split: train={len(result['train'])}, dev={len(result['dev'])}")
        return result
    
    return dataset


# Parse command line arguments
parser = argparse.ArgumentParser(description='Cold Start Training')
parser.add_argument('--model_path', type=str, default="/YOUR/MODEL", 
                    help='Path to the pretrained model')
parser.add_argument('--load_human_data', type=str, default='false', 
                    help='Load human data (true/false)')
args = parser.parse_args()

model_path = args.model_path
LOAD_HUMAN_DATA = args.load_human_data.lower() == 'true'
print(f"Loading model: {model_path}")
if LOAD_HUMAN_DATA:
    print(f"Load human data: {LOAD_HUMAN_DATA}")     
tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True,
        use_fast=True,
    )

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
            )
    
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    attn_implementation = 'flash_attention_2'
    )

# raw = load_dataset("json", data_files={"train": "PostTraining/ColdStart/dataset/train.jsonl", "dev": "PostTraining/ColdStart/dataset/test.jsonl"})

if LOAD_HUMAN_DATA:
    raw = load_human_dataset(split=True, train_ratio=0.9)
else:
    raw = load_dataset_from_hf("Godheritage/BesiegeField_geminidataset_coldstart", train_ratio=0.9, seed=42)
ds = raw

training_args = SFTConfig(
    output_dir="/cold_start_output_dir",
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    num_train_epochs=36,
    learning_rate=1e-6,
    bf16=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.03,
    weight_decay=0.1,
    logging_steps=1,
    save_strategy="steps",
    optim="adamw_8bit",
    save_steps=16,
    save_total_limit=400,
    report_to="tensorboard",
    ignore_data_skip=False,
    dataloader_drop_last=False,
    max_length=4736,
    eos_token=tokenizer.eos_token,
    pad_token=tokenizer.pad_token,
    use_liger_kernel=True,
)

oft_config = OFTConfig(
    r=0,
    oft_block_size=64,
    use_cayley_neumann=True,
    target_modules="all-linear",
    bias="none",
    task_type=TaskType.CAUSAL_LM
)

trainer = SFTTrainer(
    model=model_path,
    train_dataset=ds["train"],
    eval_dataset=ds["dev"] if training_args.eval_strategy != "no" else None,
    processing_class=tokenizer,
    args=training_args,
    peft_config=oft_config,
)

last_ckpt = get_last_checkpoint(training_args.output_dir)
trainer.train(resume_from_checkpoint=last_ckpt)

