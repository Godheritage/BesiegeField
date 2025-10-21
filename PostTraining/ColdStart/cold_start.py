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


# Parse command line arguments
parser = argparse.ArgumentParser(description='Cold Start Training')
parser.add_argument('--model_path', type=str, default="/YOUR/MODEL", 
                    help='Path to the pretrained model')
args = parser.parse_args()

model_path = args.model_path
print(f"Loading model: {model_path}")     
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

