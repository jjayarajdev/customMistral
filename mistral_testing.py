import torch
import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM,BitsAndBytesConfig
import warnings as wr
wr.filterwarnings('ignore')

model_name='mistralai/Mistral-7B-v0.3'
bnb_config=BitsAndBytesConfig(load_in_4bit= True,bnb_4bit_use_double_quant=True,bnb_4bit_quant_type='nf4',bnb_4bit_compute_dtype=torch.bfloat16)

model=AutoModelForCausalLM.from_pretrained(model_name,quantization_config=bnb_config,device_map='auto',trust_remote_code=True)
eval_tokenizer=AutoTokenizer.from_pretrained(model_name)
eval_tokenizer.pad_token = eval_tokenizer.eos_token

from peft import PeftModel
ft_model=PeftModel.from_pretrained(model,'sri-lasya/mistral_fine_tuned')
ft_model.eval()

def evaluation(prompt):
  model_input = eval_tokenizer(prompt, return_tensors="pt",padding=False).to("cuda")
  with torch.no_grad():
    generated_ids=ft_model.generate(**model_input,max_new_tokens=128)
    decoded_output = eval_tokenizer.batch_decode(generated_ids)

  return decoded_output[0]
request=True
while request==True:
    p=input('Enter your Question related to GST to test the model? ') 
    output=evaluation(p)
    print(output)    
    r=input('Would you like to check more values? Type Yes/No ')
    if r=='Yes' or r=='yes':
            request=True
    elif r=='No' or r=='no':
        request=False

