import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import pandas as pd
import time
import uuid
from bs4 import BeautifulSoup
import argparse
import json
import re

def preprocess(text):
    text = text.lower()
    text = text.replace('\n', ' ').replace('\t', ' ')
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

from rouge_score import rouge_scorer
from bert_score import score as bert_score
import numpy as np

argparser = argparse.ArgumentParser()
argparser.add_argument("--max", required=True)
argparser.add_argument("--ndocs", required=True)
argparser.add_argument("--ntop", required=True)
args = argparser.parse_args()
retrieve_number = int(args.ndocs)
num_reranked_docs = int(args.ntop)

data = ['python_questions0.csv', 'python_questions1.csv', 'python_questions2.csv', 'python_questions3.csv']

MAX_DOCS = int(args.max)
df = pd.DataFrame()
for d in data:
    df = pd.concat([df, pd.read_csv(d)], ignore_index=True)

# Keep only the specified columns (no row limit)
df = df.loc[:min(len(df), MAX_DOCS-1), ["tags", "question_title", "question_body", "answer", "question_score"]]
total_docs = len(df)

chunks = []
min_code_block = 10

for ix, content in df.iterrows():
    answer = content.loc['answer']
    tags = content.loc["tags"]
    score = content.loc["question_score"]

    question_chunk = f"{content.loc['question_title']}\n{content.loc['question_body']}".lower()
    chunks.append({"chunk": question_chunk,
                   "metadata": {"tags": tags,
                                "score": score,
                                "question": True,
                                "code": False,
                                "answer": answer.lower()
                                }})
    
    answer_chunk = str(answer).lower()
    chunks.append({"chunk": answer_chunk,
                   "metadata": {"tags": tags,
                   "score": score,
                   "question": False,
                   "code": False}
                   })

    soup = BeautifulSoup(answer, 'html.parser')
    code_blocks = [code.get_text() for code in soup.find_all('code')]
    for block in code_blocks:
        if len(block) > min_code_block and '\n' in block.strip():
            chunks.append({"chunk": block.lower(),
                           "metadata": {"tags": tags,
                                        "score": score,
                                        "question": False,
                                        "code": True}})

chunks = pd.DataFrame(chunks)
total_chunks = len(chunks)
print(f"Prepared {total_chunks} chunks.")


# Initialize Chroma client
client = chromadb.PersistentClient(path="./test_db")

collection = client.get_or_create_collection(
    name="stackoverflow_demo",
    metadata={"hnsw:space": "cosine"}
)

# Initialize model
model = SentenceTransformer('all-MiniLM-L6-v2')

BATCH_SIZE = 200
total_added = 0
start_time = time.time()

for batch_num in range(0, total_chunks, BATCH_SIZE):
    batch = chunks.iloc[batch_num:batch_num + BATCH_SIZE]
    
    documents = []
    metadatas = []
    ids = []
    embeddings = []
    
    for ix, row in batch.iterrows():
        chunk = row["chunk"]
        metadata = row["metadata"]
        documents.append(chunk)
        metadatas.append(metadata)
        ids.append(str(uuid.uuid4()))
        embeddings.append(model.encode(chunk).tolist())
    
    collection.add(
        documents=documents,
        metadatas=metadatas,
        ids=ids,
        embeddings=embeddings
    )
    total_added += len(documents)


from transformers import AutoTokenizer, AutoModelForCausalLM

class RAG:
    def __init__(self, embedder, collection, reranker, retrieve_number=3, num_reranked_docs=2, gpu_based=False):
        self.gpu_based = gpu_based
        model_id = "stabilityai/stablelm-2-zephyr-1_6b" if gpu_based else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.device = "cuda" if self.gpu_based else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.llm = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device)
        self.embedder = embedder
        self.retriever = collection
        self.retrieve_number = retrieve_number
        self.reranker = reranker
        self.num_reranked_docs = num_reranked_docs


    def generate(self, query):
        query_embedding = self.embedder.encode(query.lower()).tolist()
        results = self.retriever.query(query_embeddings=[query_embedding], n_results=self.retrieve_number)
        # reranking
        docs = results['documents'][0]
        metadatas = results['metadatas'][0]
        pairs = [(query, doc) for doc in docs]
        scores = self.reranker.predict(pairs)
        scores_ranked = sorted(zip(docs, metadatas, scores), key=lambda x: x[2], reverse=True)
        top_docs_metas = scores_ranked[:self.num_reranked_docs]
        # repackage top_docs_metas
        reranked_results = {
        "documents": [[doc for doc, meta, _ in top_docs_metas]],
        "metadatas": [[meta for doc, meta, _ in top_docs_metas]]
        }
        prompt = self.build_prompt(query, reranked_results)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        outputs = self.llm.generate(**inputs, max_new_tokens=200)
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return output.split("Answer:")[1]

    def context_from_results(self, results):
        contexts = []
        for document, metadata in zip(results["documents"][0], results["metadatas"][0]):
            if metadata["question"]:
                contexts.append(metadata["answer"])
            else:
                contexts.append(document)
        return contexts

    def build_prompt(self, query, results):
        contexts = self.context_from_results(results)
        return f'''
            Answer the following code related question using the context provided inside triple qoutes in it is useful.
            In the answer provide an example of code that is related to the question.
            If you do not know the answer, say that you do not know. Do not try to invent the solution.
            

            Question: {query}


            ```{''.join(f"Context {i}: {context}{chr(10)}{chr(10)}" for i, context in enumerate(contexts))}´´´

            
            Answer:

            '''
    

class BasicLLM:
    def __init__(self, gpu_based=False):
        self.gpu_based = gpu_based
        model_id = "stabilityai/stablelm-2-zephyr-1_6b" if gpu_based else "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.device = "cuda" if gpu_based else "cpu"
        self.llm = AutoModelForCausalLM.from_pretrained(model_id, device_map=self.device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        

    def generate(self, query):
        inputs = self.tokenizer(query, return_tensors="pt").to(self.device)
        outputs = self.llm.generate(**inputs, max_new_tokens=200)
        output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        if output.startswith(query):
            return output[len(query):].strip()
        return output.strip()
    

reranker = CrossEncoder('BAAI/bge-reranker-base')

code_llm = RAG(model, collection, reranker, retrieve_number, num_reranked_docs)

basicLLM = BasicLLM()

"""simple_questions = ["How does Python's if __name__ == '__main__': work?"]

simple_answers = [
    "The statement `if __name__ == '__main__':` checks whether the script is being run directly (not imported as a module). If so, the code under this block will execute."
    ]


semi_adv_instructions = ["Write code for removing all duplicate elements of a list."]

semi_adv_answers = [
    "You can remove duplicates using: `unique_list = list(set(original_list))`."
 ]

advanced_instructions = ["Plan and provide code for a weather app. Provide an implemetation plan, design, code, instructions for hosting and anything else that might be needed."]

advanced_answers = [
    "To build a weather app:\n1. **Plan**: Fetch real-time weather data using an API like OpenWeatherMap.\n2. **Design**: UI with city input, forecast display, and icons for weather types.\n3. **Code**: Use Python (Flask or FastAPI) for backend, and HTML/CSS/JavaScript or Streamlit for frontend.\n4. **Instructions**: Host on Heroku or Render. Set up API keys securely. Use requests for API calls.\n5. **Extra**: Add caching, error handling, unit tests, and optional location auto-detection."
    ]"""

simple_questions = ["How does Python's if __name__ == '__main__': work?",
                    "What is the difference between .loc[] and .iloc[] in Pandas?",
                    "How do you save a sci-kit learn model?"]

simple_answers = [
    "The statement `if __name__ == '__main__':` checks whether the script is being run directly (not imported as a module). If so, the code under this block will execute.",
    "In Pandas, `.loc[]` is label-based indexing (using row and column names), while `.iloc[]` is integer position-based indexing.",
    "You can save a scikit-learn model using `import joblib` and `joblib.dump(model, 'model.pkl'). Load it later with `joblib.load('model.pkl')"
    ]


semi_adv_instructions = ["Write code for removing all duplicate elements of a list.",
                         "Reverse a list in Python without using built-in methods?",
                         "Check for duplicates in a specific column in pandas Dataframe"]

semi_adv_answers = [
    "You can remove duplicates using: `unique_list = list(set(original_list))`.",
    "You can reverse a list manually like this:\n```python\nreversed_list = [original_list[i] for i in range(len(original_list)-1, -1, -1)]\n```",
    "Check duplicates in a column using:\n```python\nduplicates = df[df['column_name'].duplicated()]"
 ]

advanced_instructions = ["Plan and provide code for a weather app. Provide an implemetation plan, design, code, instructions for hosting and anything else that might be needed.",
                         "Write complete code for RAG pipeline. Include unit tests for each step.",
    "Can you explain backpropagation and provide a concrete teaching example?"
    ]

advanced_answers = [
    "To build a weather app:\n1. **Plan**: Fetch real-time weather data using an API like OpenWeatherMap.\n2. **Design**: UI with city input, forecast display, and icons for weather types.\n3. **Code**: Use Python (Flask or FastAPI) for backend, and HTML/CSS/JavaScript or Streamlit for frontend.\n4. **Instructions**: Host on Heroku or Render. Set up API keys securely. Use requests for API calls.\n5. **Extra**: Add caching, error handling, unit tests, and optional location auto-detection."
    "A basic RAG pipeline includes: document chunking, embedding generation (e.g., with SentenceTransformers), vector store retrieval (e.g., FAISS), and LLM integration.\nUnit tests should cover:\n- Chunking edge cases\n- Embedding shape and type checks\n- Retrieval relevance accuracy\n- Response generation fidelity using mocked LLM output.",
    "Backpropagation is how neural networks learn: it computes gradients of the loss with respect to weights by applying the chain rule layer-by-layer backward.\nExample: For a 2-layer MLP, manually compute derivatives of the loss (e.g., MSE) with respect to weights and biases using a small input/output example."]


# RAG
simple_responses = [code_llm.generate(query) for query in simple_questions]
semi_adv_responses = [code_llm.generate(query) for query in semi_adv_instructions]
advanced_responses = [code_llm.generate(query) for query in advanced_instructions]

# Basic LLM
simple_responses_basic = [basicLLM.generate(query) for query in simple_questions]
semi_adv_responses_basic = [basicLLM.generate(query) for query in semi_adv_instructions]
advanced_responses_basic = [basicLLM.generate(query) for query in advanced_instructions]


results_dict = {
    "RAG": {
        "simple": [
            {"question": q, "ground_truth": gt, "generated_answer": ans}
            for q, gt, ans in zip(simple_questions, simple_answers, simple_responses)
        ],
        "semi_advanced": [
            {"question": q, "ground_truth": gt, "generated_answer": ans}
            for q, gt, ans in zip(semi_adv_instructions, semi_adv_answers, semi_adv_responses)
        ],
        "advanced": [
            {"question": q, "ground_truth": gt, "generated_answer": ans}
            for q, gt, ans in zip(advanced_instructions, advanced_answers, advanced_responses)
        ]
    },
    "BasicLLM": {
        "simple": [
            {"question": q, "ground_truth": gt, "generated_answer": ans}
            for q, gt, ans in zip(simple_questions, simple_answers, simple_responses_basic)
        ],
        "semi_advanced": [
            {"question": q, "ground_truth": gt, "generated_answer": ans}
            for q, gt, ans in zip(semi_adv_instructions, semi_adv_answers, semi_adv_responses_basic)
        ],
        "advanced": [
            {"question": q, "ground_truth": gt, "generated_answer": ans}
            for q, gt, ans in zip(advanced_instructions, advanced_answers, advanced_responses_basic)
        ]
    }
}

SUFFIX = f"max-{MAX_DOCS}_ndocs-{retrieve_number}_ntop-{num_reranked_docs}"
# Save to json
with open(f"outputs_rag_basic_{SUFFIX}.json", "w", encoding="utf-8") as file:
    json.dump(results_dict, file, indent=4, ensure_ascii=False)


"""
reference_answers = {
    model_type: {
        category: [item["ground_truth"] for item in items]
        for category, items in results.items()
    }
    for model_type, results in results_dict.items()
}

generated_answers = {
    model_type: {
        category: [item["generated_answer"] for item in items]
        for category, items in results.items()
    }
    for model_type, results in results_dict.items()
}
"""


def rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rougeL'], use_stemmer=True)
    for pred, ref in zip(predictions, references):
        scores = scorer.score(ref, pred)
        yield scores['rouge1'].fmeasure, scores['rougeL'].fmeasure

def bert(predictions, references):
    P, R, F1 = bert_score(predictions, references, lang='en', rescale_with_baseline=True)
    return P.mean().item(), R.mean().item(), F1.mean().item()

def evaluation(predictions, references):
    rouge_scores = list(rouge(predictions, references))
    # mean rouge results
    rouge1_f1 = np.mean([score[0] for score in rouge_scores])
    rougeL_f1 = np.mean([score[1] for score in rouge_scores])
    
    bert_scores = bert(predictions, references)
    bert_precision, bert_recall, bert_f1 = bert_scores
    
    # evaluate on all questions
    
    return {
        "rouge1": rouge1_f1,
        "rougeL": rougeL_f1,
        "bert_precision": bert_precision,
        "bert_recall": bert_recall,
        "bert_f1": bert_f1
    }

evaluation_results = {}
for model_type, categories in results_dict.items():
    model_evaluation = {}
    for category, items in categories.items():
        references = [item["ground_truth"] for item in items]
        predictions = [preprocess(item["generated_answer"]) for item in items]
        model_evaluation[category] = evaluation(predictions, references)
    evaluation_results[model_type] = model_evaluation

# save evaluation results
with open(f"evaluation_results_rag_basic_{SUFFIX}.json", "w", encoding="utf-8") as file:
    json.dump(evaluation_results, file, indent=4, ensure_ascii=False)

total_evaluation_results = {}
for model_type, categories in evaluation_results.items():
    for category, items in categories.items():
        if category not in total_evaluation_results:
            total_evaluation_results[category] = {
                "rouge1": 0.0,
                "rougeL": 0.0,
                "bert_precision": 0.0,
                "bert_recall": 0.0,
                "bert_f1": 0.0
            }
        for key, value in items.items():
            total_evaluation_results[category][key] += value / len(evaluation_results)

# save evaluation results
with open(f"total_evaluation_results_{SUFFIX}.json", "w", encoding="utf-8") as file:
    json.dump(total_evaluation_results, file, indent=4, ensure_ascii=False)
