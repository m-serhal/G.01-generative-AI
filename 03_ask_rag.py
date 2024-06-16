# This script can run both locally (w/LM Studio) or with an OpenAI key.
from openai import OpenAI
import numpy as np
import json
from config import *

embeddings_json= "knowledge_pool/merged_output.json"

# Choose between "local" or "openai" mode
mode = "local" # or "local"
client, completion_model = api_mode(mode)

# question = "What is the program for the building?"
# question = "What is the place like?"
# question = "Is there any mention of the construction materials that should be used?"
question = "The image features a modern building with a contemporary design. The facade is predominantly made of glass and metal, creating an aesthetically pleasing and reflective surface. The use of these materials allows for significant energy savings as they have low heat absorption properties. \n\nThe glass can be tinted or coated to reduce the amount of sunlight entering the building, thus minimizing energy consumption during hotter periods of the year. Additionally, the metal used in the construction can be made of recyclable materials, reducing the environmental impact during the building's life cycle. \n\nAnother important factor is the integration of efficient lighting systems inside the building. Proper lighting not only enhances the building's functionality but also reduces energy consumption by utilizing LED lights or other energy-efficient solutions. This can lead to substantial energy savings and contribute to a more sustainable environment.\n\nFurthermore, the building can be designed with features such as solar panels, green roofs, or other renewable energy systems to further reduce its carbon footprint. By incorporating these technologies, the building can harness renewable energy sources and become more self-sufficient in terms of power production.\n\nIn conclusion, the image showcases a building that is not only aesthetically pleasing with its contemporary design but also highly functional and environmentally friendly. It demonstrates how the use of modern materials like glass and metal, as well as the integration of renewable energy solutions, can contribute to a more sustainable urban landscape check if this prompt comply with LEED values"


num_results = 20 #how many vectors to retrieve


def get_embedding(text, model=embedding_model):
    text = text.replace("\n", " ")
    response = local_client.embeddings.create(input = [text], model=model)
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
            {"role": "system", 
             "content": prompt
            },
            {"role": "user", 
             "content": question
            }
        ],
        temperature=0.1,
    )
    return completion.choices[0].message.content

print("Waiting for an answer...")
# Embed our question
question_vector = get_embedding(question)

# Load the knowledge embeddings
index_lib = load_embeddings(embeddings_json)

# Retrieve the best vectors
scored_vectors = get_vectors(question_vector,index_lib)
scored_contents = [vector['content'] for vector in scored_vectors]
rag_result = "\n".join(scored_contents)

# Get answer from rag informed agent
prompt = f"""Answer the question based on the provided information. 
            You are given the extracted parts of a long document and a question. Provide a direct answer.
            If you don't know the answer, just say "I do not know.". Don't make up an answer.
            PROVIDED INFORMATION: """ + rag_result

print(prompt)
answer = rag_answer(question, prompt)
print("ANSWER:")
print(answer)
