### What is language modeling?
Language modeling (LM) is the use of various statistical and probabilistic techniques to determine the probability of a given sequence of words occurring in a sentence. It is the task of fitting a model to a corpus, which can be domain specific. Language models analyze bodies of text data to provide a basis for their word predictions. They are widely
used in natural language processing (NLP) applications, particularly ones that generate text as an output.

Most popular transformer-based models are trained using a variant of language modeling, e.g. BERT with masked language modeling, GPT-2 with causal language modeling. And they are two basic language modeling tasks. 


### Masked Language Modeling  

Masked language modeling is the task of masking tokens in a sequence with a masking token, and prompting the model to fill that mask with an appropriate token. This allows the model to attend to both the right context (tokens on the right of the mask) and the left context (tokens on the left of the mask). Such a training creates a strong basis for downstream tasks requiring bi-directional context, such as question answering. 

### Causal Language Modeling
Causal language modeling is the task of predicting the token following a sequence of tokens. In this situation, the model only attends to the left context (tokens on the left of the mask). Such a training is particularly interesting for generation tasks. Usually, the next token is predicted by sampling from the logits of the last hidden state the model produces from the input sequence.

Language modeling can be useful outside of pretraining as well, for example to shift the model distribution to be domain-specific: using a language model trained over a very large corpus, and then fine-tuning it to a news dataset or on scientific papers.

### Start to train
These notebooks will teach you how to fine-tune a [towhee transformers operator](https://towhee.io/text-embedding/transformers) in both language modeling tasks using [hugging face transformer Trainer](https://huggingface.co/docs/transformers/main_classes/trainer) backend.
- For Masked Language Modeling, go to [fine_tune_bert_on_masked_language_modeling.ipynb](./fine_tune_bert_on_masked_language_modeling.ipynb)
- For Causal Language Modeling, go to [fine_tune_gpt2_on_causal_language_modeling.ipynb](./fine_tune_gpt2_on_causal_language_modeling.ipynb)
