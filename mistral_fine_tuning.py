from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments,Trainer,DataCollatorForLanguageModeling
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training
from sklearn.model_selection import train_test_split
from datasets import load_dataset,Dataset
from trl import SFTTrainer
import torch
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from evaluate import evaluator
import warnings as wr
wr.filterwarnings('ignore')

train_dataset=load_dataset('json',data_files='dataset.jsonl',split='train')

eval_dataset = load_dataset('json',data_files='test.jsonl',split='train')

model_name='mistralai/Mistral-7B-v0.3'
nf4_config = BitsAndBytesConfig(load_in_4bit=True,bnb_4bit_quant_type='nf4',bnb_4bit_use_double_quant=True,bnb_4bit_compute_dtype=torch.bfloat16)

model= AutoModelForCausalLM.from_pretrained(model_name,device_map='cuda:0',quantization_config=nf4_config,use_cache=False)
tokenizer = AutoTokenizer.from_pretrained(model_name,add_bos_token=True,add_eos_token=True,padding_side='left')
tokenizer.pad_token = tokenizer.eos_token

def formatting_func(example):
    text = f"### Question: {example['Question']}\n ### Answer: {example['Answer']}"
    return text

max_length = 2048

def generate_and_tokenize_prompt(prompt):
    result = tokenizer(formatting_func(prompt),truncation=True,max_length=max_length, padding="max_length")
    result["labels"] = result["input_ids"].copy()
    return result

tokenized_train_dataset = train_dataset.map(generate_and_tokenize_prompt)
tokenized_val_dataset = eval_dataset.map(generate_and_tokenize_prompt)
def print_trainable_parameters(model):
    trainable_params=0
    all_params=0
    for i,param in model.named_parameters():
        all_params+=param.numel()
        if param.requires_grad:
            trainable_params+=param.numel()
    print(f"Trainable paramereters : {trainable_params} || all params: {all_params} || trainable%: {100*trainable_params/all_params}")

peft_config=LoraConfig(lora_alpha=16,lora_dropout=0.1,r=64,bias='none',task_type='CAUSAL_LM',target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj','lm_head'])
model=prepare_model_for_kbit_training(model)
model=get_peft_model(model,peft_config)
print_trainable_parameters(model)

if torch.cuda.device_count() > 1:
    model.is_parallelizable =True
    model.model_parallel =True

args=TrainingArguments(output_dir='mistral_fine_tuned',
                       max_steps=100,
                       per_device_train_batch_size=15,
                       warmup_steps=0.03,
                       gradient_accumulation_steps=1,
                       gradient_checkpointing=True,
                       optim='paged_adamw_8bit',                    
                       logging_steps=10,
                       save_strategy='steps',
                       save_steps=25,
                       logging_dir='./logs',
                       evaluation_strategy='steps',
                       do_eval=True,
                       learning_rate=2e-5,
                       bf16=True,
                       lr_scheduler_type='constant')

trainer = Trainer(
  model=model,
  tokenizer=tokenizer,
  args=args,
  train_dataset=tokenized_train_dataset,
  eval_dataset=tokenized_val_dataset,
  data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False))

model.config.use_cache=False
trainer.train()
trainer.save_model('mistral_fine_tuned')

eval_results = trainer.evaluate(eval_dataset=tokenized_val_dataset)
validation_loss = eval_results["eval_loss"]
trainer.push_to_hub('sri-lasya/gst-taxing-llm')

merged_model=model.merge_and_unload()

