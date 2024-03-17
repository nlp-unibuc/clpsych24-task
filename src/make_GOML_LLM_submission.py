#####################################################
#   Good Old-fashioned Machine Learning Pipeline -- submission 2
#   1. train on test set
#   2. use shap to explain the expert set
#    highlights consist of entire sentences where important features are discovered
#   3. use LLM to get a better summary of evidence
#####################################################


import pandas as pd
import shap
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from nltk import ngrams
from nltk.tokenize import word_tokenize
import re
import os
import csv
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain_community.llms import LlamaCpp

from summarization import summarize
from users2postid import all_user_ids_to_post_ids
from data_loaders import *



def verify_unique_strings(strings):
    unique_strings = []

    for string in strings:
        string_copy = string
        tokens = word_tokenize(string_copy)
        substrings = [' '.join(gram) for gram in ngrams(tokens, 3) if all(len(word) >= 3 for word in gram)]

        is_unique = all(
            all(sub not in other_string and other_string not in sub for other_string in unique_strings) for sub in
            substrings)

        if is_unique:
            unique_strings.append(string)

    return unique_strings


def extract_sentence_with_feature(post_body, feature):
    post_body_copy = post_body
    post_body_copy = clean_sentence(post_body_copy)
    tokens = word_tokenize(post_body_copy)
    sentences = [sentence.strip() for sentence in re.split(r'[.!?]', post_body_copy) if
                 feature in sentence]
    return sentences


def clean_sentence(sentence):
    cleaned_sentence = re.sub(r'[^A-Za-z0-9\s.,\'"!?-_[]()]', '', sentence)
    return cleaned_sentence


def get_summarized_evidence(post_id, shap_values, dataset, ft_names):
    matching_rows = dataset[dataset['post_id'] == post_id]

    if not matching_rows.empty:
        post_index = matching_rows.index[0]

        if post_index < len(shap_values):
            post_shap_values = shap_values[post_index]
            top_feature_indices = np.argsort(post_shap_values)[:50]
            top_features = ft_names[top_feature_indices]
            unique_top_features = verify_unique_strings(top_features)

            summarized_evidence = []
            post_body = dataset.loc[post_index, 'post_body']

            for feature in unique_top_features:
                sentences = extract_sentence_with_feature(post_body, feature)
                summarized_evidence.extend(sentences)

            summarized_evidence = list(set(summarized_evidence))

            return summarized_evidence
    else:
        return []


def verify_top_features_existence(unique_top_features, post_body):
    verified_top_features = []

    for feature in unique_top_features:
        if feature.lower() in post_body.lower():
            verified_top_features.append(feature)

    return verified_top_features


def get_highlights(post_id, shap_values, dataset, ft_names):
    matching_rows = dataset[dataset['post_id'] == post_id]

    if not matching_rows.empty:
        post_index = matching_rows.index[0]

        if post_index < len(shap_values):
            post_shap_values = shap_values[post_index]
            top_feature_indices = np.argsort(post_shap_values)[:50]
            top_features = ft_names[top_feature_indices]
            unique_top_features = verify_unique_strings(top_features)

            highlights = []
            post_body = dataset.loc[post_index, 'post_body']

            for feature in unique_top_features:
                sentences = extract_sentence_with_feature(post_body, feature)
                highlights.extend(sentences)

            highlights = list(set(highlights))

            return highlights
    else:
        return []

def generate_json_output(user_id, post_id, shap_values, dataset, ft_names, llm_model):

    user_data = {
        "summarized_evidence": "",
        "posts": [
            {
                "post_id": post_id,
                "highlights": ""
            }
        ],
        "optional": [
            []
        ]
    }

    return {user_id: user_data}





data_mapping_expert = load_csv1(path_find1('umd_reddit_suicidewatch_dataset_v2', 'expert'))
expert_posts=data_mapping_expert['posts']
expert_users=data_mapping_expert['users']
expert_users=[user for user in expert_users if not user.get('label', '') == '']
expert_users=[user for user in expert_users if not user.get('label', '') == 'a']
expert_posts=clustering_data1(expert_posts,expert_users)
data = pd.DataFrame(expert_posts)                     #cei 209 de experti


data_mapping_test = load_csv2(path_find2('umd_reddit_suicidewatch_dataset_v2', 'crowd', 'test'), 'test')
shared_test_posts_data = data_mapping_test['shared_task_posts']
crowd_test_data = data_mapping_test['crowd_train']
task_A_test_data = data_mapping_test['task_A_train']
data_test = clustering_data2(task_A_test_data, crowd_test_data, shared_test_posts_data)
df_test = pd.DataFrame(data_test)


mapare = {"a": 1, "b": -1, "c": -1, "d": -1}
df_test['label'] = df_test['label'].apply(lambda x: mapare[x])
data['label'] = data['label'].apply(lambda x: mapare[x])

tfidf_vectorizer = TfidfVectorizer(**{
    'min_df': 1,
    'max_features': None,
    'strip_accents': 'unicode',
    'analyzer': 'word',
    'token_pattern': r'\b[^\d\W]+\b',
    'ngram_range': (2, 4),
    'use_idf': True,
    'smooth_idf': True,
    'sublinear_tf': True,
    # 'vocabulary': all_keywords,
    # 'stop_words': stop_words#,
})

test_features = tfidf_vectorizer.fit_transform(df_test['post_title'] + df_test['post_body'])

model = LogisticRegression(class_weight='balanced')
model.fit(test_features, df_test['label'])

callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

llm_model = LlamaCpp(
    model_path="openhermes-2.5-mistral-7b.Q4_K_M.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    n_ctx=32000,
    callback_manager=callback_manager,
    verbose=True,
)



new_data = tfidf_vectorizer.transform(data['post_title'] + data['post_body']).toarray()
explainer_new = shap.LinearExplainer(model, new_data, feature_dependence="independent")
shap_values_new = explainer_new.shap_values(new_data)

feature_names_new = np.array(tfidf_vectorizer.get_feature_names_out())

output_data_new = {}

for user_id, user_posts in data.head(-1).groupby('user_id'):
    user_data = {
        "summarized_evidence": "",
        "posts": [],
        "optional": [[]]
    }

    text = ""

    for index, row in user_posts.iterrows():
        post_id = row['post_id']

        extracted_sentences = get_summarized_evidence(post_id, shap_values_new, data, feature_names_new)

        text += ". ".join(extracted_sentences) + " " if extracted_sentences else ""

        user_data["posts"].append({
            "post_id": post_id,
            "highlights": get_highlights(post_id, shap_values_new, data, feature_names_new)
        })

    question = f"You are a licensed psychologist and expert therapist evaluating a case. Analyze the following text and make a professional summary of the content explaining why the individual has suicidal thoughts. The text is as follows: {text}"
    llm_result = llm_model(question) #llm_model(' '.join(question.split(' ')[:300]))

    find = llm_result.find("\n\n")
    if find != -1:
        llm_result = llm_result[find + 2:]
    llm_result = llm_result.replace("\n\n", '')

    user_data["summarized_evidence"] = llm_result
    output_data_new[user_id] = user_data

users_to_keep = set(all_user_ids_to_post_ids.keys())

users_to_delete = [user_id for user_id in output_data_new.keys() if user_id not in users_to_keep]

for user_id in users_to_delete:
    del output_data_new[user_id]

output_file_path_new = 'GOML_LLM_V1.json'
with open(output_file_path_new, 'w', encoding='utf-8') as json_file:
    json.dump(output_data_new, json_file, indent=2, ensure_ascii=False)

print(f"Output saved to {output_file_path_new}")

output_file_path_new = 'GOML_LLM_V1.json'

with open(output_file_path_new, 'r', encoding='utf-8') as json_file:
    json_data = json.load(json_file)
print(json.dumps(json_data, indent=2)[:1000])


