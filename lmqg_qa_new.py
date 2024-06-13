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

def document_splitter(text): 
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=1)
    chunk = splitter.split_text(text)
    return chunk

def qa_dataset_constructor(text):
    model=TransformersQG(language='en',model='lmqg/t5-base-squad-qag')
    qa_pairs=[]
    for context in text:
        qa_pairs.append(model.generate_qa(context))
    qa_pairs= list(chain.from_iterable(qa_pairs))
    # Convert list of tuples to a list of dictionaries
    qa_list_of_dicts = [{"Question": question, "Answer": answer} for question, answer in qa_pairs]
    return qa_list_of_dicts

file_text='taxation_data_updated.pdf'
text=extract_file(file_text)
text=document_splitter(text)
qa_list_of_dicts=qa_dataset_constructor(text)


# Write the training data to a JSONL file
with open("data_new.jsonl", "w") as train_file:
    for item in qa_list_of_dicts:
        train_file.write(json.dumps(item) + "\n")

