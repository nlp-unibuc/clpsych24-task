#####################################################
#   Good Old-fashioned Machine Learning Pipeline -- version 2
#   1. train on All data Expert + Task A set (train+test)
#   2. use shap to explain the expert set
#    highlights consist of a context window of 14 words before and after each matched
#    feature, not exceeding the sentence boundary
#   3. use LLM to get a better summary of evidence
#####################################################

from pprint import pprint
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
from tqdm import tqdm



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





def verify_unique_strings_highlights(strings):
    unique_strings = set()

    for string in strings:
        tokens = word_tokenize(string.lower())

        valid_tokens = [word for word in tokens if len(word) >= 3]

        reconstructed_string = ' '.join(valid_tokens)
        if reconstructed_string not in unique_strings:
            if any(contraction in string.lower() for contraction in [" ve ", " re ", " ll ", " m ", " t ", " s "]):
                reconstructed_string = re.sub(r'\b(ve|re|ll|m)\b', lambda match: "'" + match.group(1),
                                              reconstructed_string)
                reconstructed_string = reconstructed_string.replace(" '", "'")
            if reconstructed_string.strip():
                unique_strings.add(reconstructed_string)

    return list(unique_strings)


def find_N_spaces_back(text, position, N):
    spaces = 0
    for i in range(position - 1, -1, -1):
        if text[i] == ' ':
            spaces += 1
            if spaces == N:
                return i+1
        # break on end of sentence
        elif text[i] in {'.', '!', '?'}:
            return i+1
    return 0

def find_N_spaces_forward(text, position, N):
    spaces = 0
    for i in range(position, len(text)):
        if text[i] == ' ':
            spaces += 1
            if spaces == N:
                return i
        elif text[i] in {'.', '!', '?'}:
            return i
    return len(text)

def remove_duplicate_highlights(highlights):
    sorted_highlights = sorted(highlights, key=len, reverse=False)
    for i in range(len(sorted_highlights)-1):
        for j in range(i + 1, len(sorted_highlights)):
            if sorted_highlights[i] in sorted_highlights[j]:
                sorted_highlights[i] = ""
                break
    return [highlight for highlight in sorted_highlights if highlight]


def verify_top_features_existence(unique_top_features, post_body):
    verified_top_features = []

    for feature in unique_top_features:
        if feature.lower() in post_body.lower():
            verified_top_features.append(feature)

    return verified_top_features


def get_highlights(post_id, shap_values, dataset, ft_names, context_words=15):
    matching_rows = dataset[dataset['post_id'] == post_id]

    if not matching_rows.empty:
        post_index = matching_rows.index[0]

        if post_index < len(shap_values):
            post_shap_values = shap_values[post_index]
            top_feature_indices = np.argsort(post_shap_values)[:20]
            top_features = ft_names[top_feature_indices]
            txt = dataset.loc[post_index, 'post_body']
            unique_top_features = verify_unique_strings_highlights(top_features)
            verified_top_features = verify_top_features_existence(unique_top_features, txt)

            alternative = verified_top_features
            highlights = []

            for feature in verified_top_features:
                feature_lower = feature.lower()
                feature_index = txt.lower().find(feature_lower)

                if feature_index != -1:
                    start_index = find_N_spaces_back(txt, feature_index, context_words)
                    end_index = find_N_spaces_forward(txt, feature_index + len(feature_lower), context_words)
                    #start_index = max(0, feature_index - context_words * 2)
                    #end_index = min(len(dataset.loc[post_index, 'post_body']),
                    #                feature_index + len(feature_lower) + context_words * 2)
                    context_text = txt[start_index:end_index].strip()
                    if context_text.lower() in txt.lower() and context_text.strip():
                        highlights.append(context_text)
            highlights = remove_duplicate_highlights(highlights)
            if "" in alternative:
                alternative.remove("")

            if highlights:
                return highlights
            elif alternative:
                return alternative
            else:
                return []
        else:
            return []
    else:
        return []



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
print(len(df_test), ' total test size')

data_mapping_train = load_csv2(path_find2('umd_reddit_suicidewatch_dataset_v2', 'crowd', 'train'))
shared_task_posts_data = data_mapping_train['shared_task_posts']
crowd_train_data = data_mapping_train['crowd_train']
task_A_train_data = data_mapping_train['task_A_train']
data_train = clustering_data2(task_A_train_data, crowd_train_data, shared_task_posts_data)
df_train = pd.DataFrame(data_train)

df_test = pd.concat([df_test,df_train])
print(len(df_test), ' total total size')
#
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

all_df_task_a = df_test['post_title'] + df_test['post_body']
all_df_experts = data['post_title'] + data['post_title']
all_data = pd.concat([all_df_task_a, all_df_experts])
all_labels = pd.concat([df_test['label'],data['label']])
features = tfidf_vectorizer.fit_transform(all_data)

print('total shape ', features.shape)

model = LogisticRegression(class_weight='balanced')
model.fit(features, all_labels)

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
idx = 0
for user_id, user_posts in data.head(-1).groupby('user_id'):
    print(user_id, " ############# ", idx)
    idx+=1
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

        highlights = get_highlights(post_id, shap_values_new, data, feature_names_new)
        print('found highlights ')
        for h in highlights:
            print('\n', h)
        user_data["posts"].append({
            "post_id": post_id,
            "highlights": highlights,
        })

    #question = f"As a psychologist and expert therapist, summarize the content with maximum 300 words by identifying any indications of suicidal thoughts. Provide evidence from the text to support your analysis.\n\nPost Body: {post_body} \n\n Analysis:"
    question = f"You are a licensed psychologist and expert therapist evaluating a case. Analyze the following text and make a professional summary of the content explaining why the individual has suicidal thoughts. \n\nThe text is as follows: {text} \n\nAnalysis:"
    llm_result = llm_model(question)

    #find = llm_result.find("\n\n")
    #if find != -1:
    #    llm_result = llm_result[find + 2:]
    #llm_result = llm_result.replace("\n\n", '')

    user_data["summarized_evidence"] = llm_result
    output_data_new[user_id] = user_data

users_to_keep = set(all_user_ids_to_post_ids.keys())

users_to_delete = [user_id for user_id in output_data_new.keys() if user_id not in users_to_keep]

for user_id in users_to_delete:
    del output_data_new[user_id]

output_file_path_new = 'tf_and_llm.json'
with open(output_file_path_new, 'w', encoding='utf-8') as json_file:
    json.dump(output_data_new, json_file, indent=2, ensure_ascii=False)

print(f"Output saved to {output_file_path_new}")

output_file_path_new = 'tf_and_llm.json'

with open(output_file_path_new, 'r', encoding='utf-8') as json_file:
    json_data = json.load(json_file)
print(json.dumps(json_data, indent=2)[:10000])


