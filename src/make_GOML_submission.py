#####################################################
#   Good Old-fashioned Machine Learning Pipeline - submission 1
#####################################################


import shap
import json
import numpy as np
from nltk import ngrams
from nltk.tokenize import word_tokenize
import re
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

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



def limit_words(text, word_limit):
    words = text.split()
    if len(words) <= word_limit:
        return text
    else:
        return ' '.join(words[:word_limit])


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


def generate_json_output(user_id, post_id, shap_values, dataset, ft_names, word_limit=300):
    extracted_sentences = get_summarized_evidence(post_id, shap_values, dataset, ft_names)

    user_data = {
        "summarized_evidence": "",
        "posts": [
            {
                "post_id": post_id,
                "highlights": get_highlights(post_id, shap_values, dataset, ft_names)
            }
        ],
        "optional": [
            []
        ]
    }

    for sentence in extracted_sentences:
        user_data["summarized_evidence"] += sentence + " "

    user_data["summarized_evidence"] = limit_words(user_data["summarized_evidence"], word_limit)

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
## Train only on the test set :o :0 :O ;)
## apparently this works equally well
model.fit(test_features, df_test['label'])
explainer = shap.LinearExplainer(model, test_features, feature_dependence="independent")
shap_values = explainer.shap_values(test_features)
X_test_array = test_features.toarray()
feature_names = np.array(tfidf_vectorizer.get_feature_names_out())


## Explain predictions on expert data
new_data = tfidf_vectorizer.transform(data['post_title'] + data['post_body']).toarray()
explainer_new = shap.LinearExplainer(model, new_data, feature_dependence="independent")
shap_values_new = explainer_new.shap_values(new_data)

feature_names_new = np.array(tfidf_vectorizer.get_feature_names_out())

output_data_new = {}

for user_id, user_posts in tqdm(data.groupby('user_id')):
    user_data = {
        "summarized_evidence": "",
        "posts": [],
        "optional": [[]]
    }

    for index, row in user_posts.iterrows():
        post_id = row['post_id']

        extracted_sentences = get_summarized_evidence(post_id, shap_values_new, data, feature_names_new)

        user_data["summarized_evidence"] += ". ".join(extracted_sentences) + " " if extracted_sentences else ""

        user_data["posts"].append({
            "post_id": post_id,
            "highlights": get_highlights(post_id, shap_values_new, data, feature_names_new)
        })
    user_data["summarized_evidence"] = summarize(user_data["summarized_evidence"])

    output_data_new[user_id] = user_data

users_to_keep = set(all_user_ids_to_post_ids.keys())

users_to_delete = [user_id for user_id in output_data_new.keys() if user_id not in users_to_keep]

for user_id in users_to_delete:
    del output_data_new[user_id]

output_file_path_new = 'GOML_submission.json'
with open(output_file_path_new, 'w', encoding='utf-8') as json_file:
    json.dump(output_data_new, json_file, indent=2, ensure_ascii=False)

print(f"Output saved to {output_file_path_new}")
output_file_path_new = 'GOML_submission.json'

with open(output_file_path_new, 'r', encoding='utf-8') as json_file:
    json_data = json.load(json_file)

print(json.dumps(json_data, indent=2)[:100])


