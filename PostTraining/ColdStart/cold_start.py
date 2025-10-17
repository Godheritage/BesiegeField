import json, os, torch
from datasets import load_dataset
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


model_path = "/YOUR/MODEL"     
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

raw = load_dataset("json", data_files={"train": "train.jsonl", "dev": "test.jsonl"})
ds = raw

training_args = SFTConfig(
    output_dir="/output_dir",
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

