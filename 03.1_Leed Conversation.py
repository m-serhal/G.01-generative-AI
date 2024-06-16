import json
import numpy as np
from openai import OpenAI
from config import *

# File path to the merged JSON containing embeddings
embeddings_json = "knowledge_pool/merged_output.json"

# Initialize OpenAI client and choose mode ("local" or "openai")
mode = "local"  # or "openai"
client, completion_model = api_mode(mode)

# Sample questions for the interview
questions = [
    "What criteria should be considered for site selection to achieve LEED Platinum certification?",
    "How can we ensure our building meets the highest water efficiency standards?",
    "What strategies can we implement to optimize energy performance for LEED Platinum?",
    "What are the best practices for selecting materials and resources to maximize LEED points?",
    "How can we improve indoor environmental quality to meet LEED Platinum standards?",
    "What innovative design features can contribute to achieving LEED Platinum certification?",
    "What documentation is required for the LEED Platinum certification process?",
    "How can we integrate renewable energy sources to enhance our LEED score?",
    "What are the key components of sustainable site development for LEED Platinum?",
    "How can we maximize the use of recycled and locally sourced materials?"

]

num_results = 20  # Number of vectors to retrieve

def get_embedding(text, model=embedding_model):
    text = text.replace("\n", " ")
    response = local_client.embeddings.create(input=[text], model=model)
    vector = response.data[0].embedding
    return vector

def similarity(v1, v2):
    return np.dot(v1, v2)

def load_embeddings(embeddings_json):
    with open(embeddings_json, 'r', encoding='utf8') as infile:
        return json.load(infile)

def get_vectors(question_vector, index_lib):
    scores = []
    for vector in index_lib:
        score = similarity(question_vector, vector['vector'])
        scores.append({'content': vector['content'], 'score': score})

    scores.sort(key=lambda x: x['score'], reverse=True)
    best_vectors = scores[0:num_results]
    return best_vectors

def rag_answer(question, prompt, model=completion_model[0]["model"]):
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": question}
        ],
        temperature=0.1,
    )
    return completion.choices[0].message.content

print("Starting a conversation with LEED consultant...")

# Load the knowledge embeddings
index_lib = load_embeddings(embeddings_json)

# Iterate over each question and get RAG responses
for idx, question in enumerate(questions, 1):
    print(f"\nQuestion {idx}: {question}")
    
    # Embed the question
    question_vector = get_embedding(question)
    
    # Retrieve the best vectors related to the question
    scored_vectors = get_vectors(question_vector, index_lib)
    scored_contents = [vector['content'] for vector in scored_vectors]
    rag_result = "\n".join(scored_contents)
    
    # Construct the RAG prompt
    prompt = f"""Answer the question based on the provided information. 
                You are given the extracted parts of a long document and a question. Provide a direct answer.
                If you don't know the answer, just say "I do not know.". Don't make up an answer.
                PROVIDED INFORMATION: {rag_result}"""
    
    # Get RAG response
    answer = rag_answer(question, prompt)
    print(f"Answer: {answer}")

print("\nend call.")
