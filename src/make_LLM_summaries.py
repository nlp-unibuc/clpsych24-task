#####################################################
#   Generate LLM summaries in a directory; run this multiple times
#####################################################

import csv
import json
import re
import sys
from tqdm import tqdm

import pandas as pd
import os
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp

from json_validator_new import all_user_ids_to_post_ids

from users2postid import all_user_ids_to_post_ids
from data_loaders import *

data_mapping_expert = load_csv1(path_find1('umd_reddit_suicidewatch_dataset_v2', 'expert'))
expert_posts=data_mapping_expert['posts']
expert_users=data_mapping_expert['users']
expert_users=[user for user in expert_users if not user.get('label', '') == '']
expert_users=[user for user in expert_users if not user.get('label', '') == 'a']
expert_posts=clustering_data1(expert_posts,expert_users)
data = pd.DataFrame(expert_posts)                     #cei 209 de experti


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
#llm_chain = LLMChain(prompt=prompt, llm=llm)

if len(sys.argv) < 2:
    print('provide the first argument a directory where to save the files')

OUT_DIR=sys.argv[1]
USE_CACHE=True

if not os.path.exists(OUT_DIR):
    os.makedirs(OUT_DIR)


summary = []
for idx,post in enumerate(tqdm(expert_posts)):
    print(f'########### summary {idx} ... {len(expert_posts)} #############')
    data = {}
    user_id = post['user_id']
    post_body = post['post_body']
    post_id = post['post_id']
    out_file = os.path.join(OUT_DIR,f'{user_id}_{post["post_id"]}.sum')
    if post_body == '':
        post_body = post['post_title']

    if os.path.exists(out_file) and USE_CACHE:
        print(f'Out file {out_file} exists, skipping...')
        result = open(out_file, 'r', encoding='utf-8').read()
    else:
        question = f"As a psychologist and expert therapist, summarize the content with maximum 300 words by identifying any indications of suicidal thoughts. Provide evidence from the text to support your analysis.\n\nPost Body: {post_body}"
        result = llm(question)  #llm(' '.join(question.split(' ')[:312]))
        with open(out_file, 'w', encoding='utf-8') as fout:
            fout.write(result)

    p = {}
    p['post_id'] = post['post_id']
    p['highlights'] = ''

    data[user_id] = {
        'summarized_evidence': result,
        'posts': [p],
        'optional': [[]]
    }
    summary.append(data)
    # break

with open(os.path.join(OUT_DIR, "summary.json"), "w", encoding="utf-8") as fis:
     json.dump(summary, fis, indent=4)
