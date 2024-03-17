import csv
import json
import re

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier  # Added RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
#import spacy
import os
#from spacy.tokens import Doc, Token
from sklearn.model_selection import GridSearchCV
import numpy as np
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp


import sys
from tqdm import tqdm


def load_csv(path_data):
    csv.field_size_limit(131072 * 10)

    expert_posts_columns = ['post_id', 'user_id', 'timestamp', 'subreddit', 'post_title', 'post_body']
    expert_posts_columns_file_path = os.path.join(path_data, "expert_posts.csv")


    expert_posts= []

    with open(expert_posts_columns_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=expert_posts_columns)
        next(reader, None)

        for row in reader:
            expert_posts.append(row)

    expert_columns = ['user_id', 'label']
    expert_path = os.path.join(path_data, "expert.csv")


    expert = []

    with open(expert_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=expert_columns)
        next(reader, None)

        for row in reader:
            expert.append(row)
    return {
        'posts': expert_posts,
        'users': expert,
    }

def path_find(file, path_data):
    path_data_set = os.getcwd()
    path_data_set = os.path.join(path_data_set, file)
    path_data_set = os.path.join(path_data_set, path_data)
    return path_data_set


def clustering_data(expert_posts, expert_users):
    data=[]
    for user in expert_users:
        user_id=user['user_id']
        for post in expert_posts:
            if post['user_id']==user_id and post['subreddit']=='SuicideWatch':
                post['label']=user['label']
                data.append(post)
    print(len(data))
    return data

data_mapping_expert = load_csv(path_find('umd_reddit_suicidewatch_dataset_v2', 'expert'))
expert_posts=data_mapping_expert['posts']
expert_users=data_mapping_expert['users']
expert_users=[user for user in expert_users if not user.get('label', '') == '']
expert_users=[user for user in expert_users if not user.get('label', '') == 'a']
expert_posts=clustering_data(expert_posts,expert_users)
data=pd.DataFrame(expert_posts)

from json_validator_new import all_user_ids_to_post_ids
data_to_predict_on = data[data.user_id.astype(str).isin(all_user_ids_to_post_ids.keys())]
data_to_predict_on.to_dict('records')
expert_posts = data_to_predict_on.to_dict(orient='records')


template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    #model_path="Nous-Hermes-2-Mixtral-8x7B-DPO.Q4_K_S.gguf",
    model_path="openhermes-2.5-mistral-7b.Q4_K_M.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    n_ctx=32000,
    callback_manager=callback_manager,
    verbose=True  # Verbose is required to pass to the callback manager
    #grammar_path='./list.gbnf'
)
# there might be an issue with the chain
# https://github.com/abetlen/llama-cpp-python/issues/944
llm_chain = LLMChain(prompt=prompt, llm=llm)

if len(sys.argv) < 2:
    print('provide the first argument a directory where to save the files')

OUT_DIR=sys.argv[1]
USE_CACHE=True

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


highlights=[]
for idx,post in enumerate(tqdm(expert_posts)):
    print(f'########### highlight {idx} ... {len(expert_posts)} #############')
    user_id= post['user_id']
    post_body = post['post_body']
    post_id = post['post_id']
    out_file = os.path.join(OUT_DIR, f'{user_id}_{post_id}.hgh')
    if post_body=='':
        post_body = post['post_title']
    if os.path.exists(out_file) and USE_CACHE:
        print(f'Out file {out_file} exists, skipping...')
        result = open(out_file, 'r', encoding='utf-8').read()
    else:
        question = f"Provide sequences of text that indicate that this person is suicidal?\n\nPost Body: {post_body}"
        result = llm_chain.run(question)
        with open(out_file, 'w', encoding='utf-8') as fout:
            fout.write(result)
    data={"user_id":user_id, "post_id":post_id, "highlights":result}
    highlights.append(data)

with open(os.path.join(OUT_DIR, "highlights.json"), "w", encoding="utf-8") as fis:
     json.dump(highlights, fis, indent=4)


