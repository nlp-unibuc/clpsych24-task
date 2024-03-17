# clpsych24-task
This repo contains the work of the UniBuc Archaeology team for CLPsych’s 2024 Shared Task. 


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
<div style="width: 100%;">
  <img width="600" src="https://github.com/nlp-unibuc/clpsych24-task/blob/main/img/pipelines.png?raw=true">
</div>


## Results

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
<div style="width: 100%;">
  <img width="600" src="https://github.com/nlp-unibuc/clpsych24-task/blob/main/img/topics.png?raw=true">
</div>


## Feature Importance
<div style="width: 100%;">
  <img width="600" src="https://github.com/nlp-unibuc/clpsych24-task/blob/main/img/violins.png?raw=true">
</div>

