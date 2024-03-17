# Cheap Ways of Extracting Clinical Markers from Texts
This repo contains the work of the UniBuc Archaeology team (Anastasia Sandu, Teodor Mihailescu, Sergiu Nisioi ) for [CLPsych’s 2024 Shared Task](https://clpsych.org/shared-task-2024/). 



## TL;DR

- a classifier paired with a machine learning explainability method can be a useful tool for identifying important sentences, phrases, and highlights that are clinically representative 
- sentences containing important features have different (statistically significant) linguistic patterns that can distinguish them from the rest
- noisy generated output containing duplicates achieves better recall and we believe that ultimately expert human judgments would be the best measure for evaluating and ranking different submissions



## Installing pre-requisities

```bash
# install requirements
pip install -r requirements.txt

# install llama-cpp
CMAKE_ARGS="-DLLAMA_BLAS=ON -DLLAMA_BLAS_VENDOR=OpenBLAS" \
  pip install llama-cpp-python

# download a spacy model for pytextrank summarization
python -m spacy download en_core_web_sm


# download the LLM
cd src && wget https://huggingface.co/TheBloke/OpenHermes-2.5-Mistral-7B-GGUF/resolve/main/openhermes-2.5-mistral-7b.Q4_K_M.gguf
```

## Pipelines
<div align="center" style="width: 100%;">
  <img width="600" src="https://github.com/nlp-unibuc/clpsych24-task/blob/main/img/pipelines.png?raw=true">
</div>

### A. Good Old-fashioned Machine Learning Pipeline

The first approach, which also obtained the highest recall amongst submissions, is based on the following steps.

#### 1. Begin with Task A
Use [crowd-annotated data](https://aclanthology.org/W19-3003/)  and map the labels to binary, i.e., assigning the label 'a' to the value -1, and the labels 'b', 'c', and 'd' to the value +1.
We cross-validate several models on [different subsamples of risk annotations](https://aclanthology.org/W18-0603/) labeled as follows: 

- **1.1 Test** - a model trained solely on Task A test set (186 posts)
- **1.2 TaskA** a model trained on the entire Task A
- **1.3 A+E** a model trained on both expert and TaskA data. 

For this downstream task of identifying highlights, we did not observe significant improvements in performance when training the logistic regression classifier with more data, nor did we observe a degradation of performance when training on the smallest amount of samples consisting only of the test set of Task A. This is encouraging for potential extensions of the GOML methodology to less-resourced languages.


#### 2. SHAP SHapley Additive exPlanations 
We use a simple linear explainer that assumes feature independence and ranks features based on a score computed as: $s_i = w_i (x_i - \hat{m}_i)$, where $w_i$ is the classifier coefficient of feature $i$, $x_i$ is the feature value in a post and $\hat{m}_i$ the mean of the feature value across all posts.

#### 3. Selecting the highlights
Requires matching the tokenized features from our tf-idf extractor to the text. 
For highlight selection, we test:

- **3.1 context window** - highlights consisting of a context window of 14 words before and after each matched feature, not exceeding the sentence boundary
- **3.2 entire sentences** - highlights consisting of entire sentences where important features are discovered in the original text

#### 4. Summarization

Here we test two options: 

- **4.1 extractive summarization** take the sentences found previously in step 3.2 and use an extractive summarization technique such as TextRank, PyTextRank to generate a summary. This method is the fastest, but performed relatively poorly, obtaining high contradiction rates (0.238) and relatively low mean consistency (0.901). 
- **4.2 LLM** achieved the best overall performance and requires taking the sentences found previously and prompting a language model to generate an abstractive summary. Our best performing system in the official ranking is configured with option 3.2 (to extract full sentences as highlights) and option 4.2 (to generate summaries using LLM).

To extract LLM summaries, we run the model only once with the following prompt: *As a psychologist and expert therapist, summarize the content by identifying any indications of suicidal thoughts. Provide evidence from the text to support your analysis. \n\n Post Body: \{content\_body\} \n\n Analysis:}.* 

The content body consists in the concatenation of important sentences instead of the post bodies. We found that the model tends to hallucinate and copy paste content from the text, unless the word \textit{Analysis} is explicitly mentioned at the end.



### B. Large Language Models
The pipeline can be summarized as follows:

- prompt the model using langchain to extract highlights from the texts for a number of $K=8$ times
- parse the LLM output and extract highlights from between quotation marks
- post-process responses: ensure the highlights are actually in the texts, remove duplicates, keep the longest matching highlights
- concatenate all posts and prompt the model without langchain to do a summary analysis of maximum 300 words


Text generation parameters are set to a temperature of 0.75, top-p nucleus sampling 1, and a maximum context size of 32000. To obtain as much data as possible, the LLM was run eight times on each post. The langchain prompt for extracting highlights is: *Provide sequences of text that indicate that this person is suicidal? \n\n Post Body: \{post\_body\}}.*

Each response is saved and post-processed to extract valid highlights present in the text, to remove duplicates, and to preserve the longest matching highlight. The model tends to be more verbose, no matter how much we change the prompt, so the post-processing step proved to be essential.



## Results

Evaluation scores of our systems in comparison to other participants in the Shared Task. The first three rows marked with superscript are the official versions we submitted during competition. The next 3 are additional experiments with highlights 3.1 or without removing duplicates and overlaps from LLM output.



| submission | recall         | precision      | recall_w       | harmonic       | consistency    | contradiction  |
|---------------------|----------------|----------------|-----------------|----------------|----------------|----------------|
| GOML            |      0.921     |      0.888     |      0.513      |      0.904     |      0.901     |      0.238     |
| GOML + LLM      | 0.939 |      0.890     |      0.390      |      0.914     | 0.973 | 0.081 |
| LLM             |      0.935     | 0.905 |  0.553 | 0.919 |      0.964     |      0.104     |
| TaskA_3.1          |      0.919     |      0.891     |      0.560      |      0.905     |      0.908     |      0.218     |
| A+E_3.1            |      0.918     |      0.892     |      0.578      |      0.905     |      0.910     |      0.217     |
| TaskA_3.1 + LLM    |      0.919     |      0.891     |      0.560      |      0.905     |      0.971     |      0.085     |
| A+E_3.1 + LLM      |      0.918     |      0.892     |  0.578 |      0.905     | 0.974 | 0.076 |
| LLM duplicates      | 0.941 | 0.907 |      0.398      | 0.924 |      0.964     |      0.104     |


## Topics
The main topics in the dataset revolve around feelings of despair, hopelessness, socioeconomic hardships, and family conflicts. Our brief analyses indicate that the texts contain strong signals for suicide and that very few subtleties can be observed in the assessment of risk degrees.

<div align="center" style="width: 100%;">
  <img width="800" src="https://github.com/nlp-unibuc/clpsych24-task/blob/main/img/topics.png?raw=true">
</div>


## Feature Importance

Given the surprising efficacy of the traditional machine learning model, we ask whether sentences containing important features have specific linguistic characteristics. Sentences are divided into two categories: **important** if they contain important features for classification and with the label **other** otherwise. 

<div align="center" style="width: 100%;">
  <img width="800" src="https://github.com/nlp-unibuc/clpsych24-task/blob/main/img/violins.png?raw=true">
</div>


Our analyses indicate that important sentences are generally more likely to have pronouns, verbs, and adjectives. In terms of mean value, pronouns and verbs are statistically different at a p-value $< 0.05$ in important sentences more often than in the rest. Similarly, mean sentence lengths are statistically larger in important sentences than in the other ones.
Adverbs show no difference between the two classes, and adjectives and nouns obtain a p-value of $0.6$ after 100,000 permutations.  

Our brief analyses show that important sentences have different (statistically significant) linguistic patterns that can distinguish them from the rest. We believe that this could be one of the reasons behind the good evaluation scores and the suitability of the GOML approach to extract highlights from this particular dataset.


