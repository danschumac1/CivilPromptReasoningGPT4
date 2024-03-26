# Prompt Ensembling for Argument Reasoning in Civil Procedures with GPT4

This project is Dan Schumacher and Dr. Anthony Rios's submission to SemEval 2024. The challenge is to identify answers to questions as correct or incorrect based on the given context. Our classification system leverages GPT4 using chain-of-thought and retrieval-augmented-generation prompt engineering, and earned us 5th place out of 21 entries. 

To obtain the datasets please follow the instructions at the link below as they are not publically available.
https://github.com/trusthlt/legal-argument-reasoning-task#obtaining-the-dataset

Due to the evolving nature of LLMs, we would like to mention that we ran all of our py scripts on the following dates

DEV			
dev_ensemble		1/30/24
few-shot		    1/28/24
few-shot_COT		1/28/24
few-shot_COT_TF		1/28/24
few-shot_RAG		1/28/24
few-shot_RAG_COT	1/07/24
few-shot_RAG_COT_TF	1/09/24
one-shot	    	1/28/24
one-shot_cot		1/30/24
one-shot_cot_TF		1/30/24

TEST
test_ensemble		1/29/24
few-shot_COT		1/29/24
few-shot_COT_TF		1/28/24
few-shot_RAG_COT	1/29/24
few-shot_RAG_COT_TF	1/29/24
one-shot		    1/29/24
one-shot_cot		1/29/24
one-shot_cot_TF		1/29/24


## Installation

To install the project, follow these steps:

1. No install is needed but you will need to put your own OpenAi .env in ./data
2. Once you have obtained the train / dev / test csvs also place them in the ./data folder