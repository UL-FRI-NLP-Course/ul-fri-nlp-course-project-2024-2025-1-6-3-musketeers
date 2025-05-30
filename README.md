# Natural language processing course: `Conversational Agent with Retrieval-Augmented Generation`

This is a repository for an agent that specializes in conversations about python code. It has integrated knowledge from Stack Overflow, so it should be able to help with answering questions regarding python and debuging the code.

# How to use it:

Firstly clone the repository
```bash
git clone https://github.com/UL-FRI-NLP-Course/ul-fri-nlp-course-project-2024-2025-1-6-3-musketeers.git
```

Make sure you have files `python_questionsX.csv` in the same directory as the file you are running (src folder). Datasets can be downloaded [from this link](https://drive.google.com/drive/folders/19JJ7XgkrGoFU80n0zLHwv-VOKniZJhFY?usp=drive_open). 

Than you can run RAG.ipynb.

At the bottom of python file, there are lists of questions that get passed to the agent. To input custom queries, you can either edit those lists, or call the agent yourself by calling `code_llm.generate(<query>)` after the `code_llm object` creation.