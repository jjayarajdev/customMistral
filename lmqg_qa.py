#importing the necessary libraries
import pdfplumber
import re
import pandas as pd
from transformers import pipeline
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from lmqg import TransformersQG
import tqdm
import json
from itertools import chain

def extract_file(file_path):
    with pdfplumber.open(file_path) as pdf:
        all_text=''
        for i in range(1,len(pdf.pages)):
            page_text=pdf.pages[i].extract_text()
            all_text+=page_text
    return all_text

def change_date_format(text):
    # Find all dates in the text
    dates = re.findall(r'\b\d{2}.\d{2}.\d{4}\b', text)
    # Replace each date with the new format
    for date in dates:
        new_date = datetime.strptime(date, '%d.%m.%Y').strftime('%d-%m-%Y')
        text = text.replace(date, new_date)
    return text

def cleaning_data(text):
    pattern=r'\b\d{1,2} \| \d{2}'
    text=re.sub(pattern,'',text)
    text=re.sub(r',','',text)
    secs=re.findall(r'\b\d{1,2}\.\d{1,2}\.\d{1,2}\.\d{1,2}',text)
    for sec in secs:
        text=text.replace(sec,',')
    secs=re.findall(r'\b\d{1,2}\.\d{1,2}\.\d{1,2} ',text)
    for sec in secs:
        text=text.replace(sec,',')
    text=text.replace('\n','')
    text=text.replace('\t','')
    return text



def document_splitter(text): 
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=1)
    chunk = splitter.split_text(text)
    return chunk

def qa_dataset_constructor(text):
    model=TransformersQG(language='en',model='lmqg/t5-base-squad-qag')
    qa_pairs=[]
    for context in tqdm(text, desc='Processing items'):
        qa_pairs.append(model.generate_qa(context))
    qa_pairs= list(chain.from_iterable(qa_pairs))
    # Convert list of tuples to a list of dictionaries
    qa_list_of_dicts = [{"Question": question, "Answer": answer} for question, answer in qa_pairs]
    return qa_list_of_dicts

file_text='taxation_data.pdf'
text=extract_file(file_text)
clean_text=change_date_format(text)
clean_text=cleaning_data(clean_text)
text=document_splitter(clean_text)
qa_list_of_dicts=qa_dataset_constructor(text)


# Write the training data to a JSONL file
with open("data.jsonl", "w") as train_file:
    for item in qa_list_of_dicts:
        train_file.write(json.dumps(item) + "\n")

