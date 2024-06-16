from config import *
from rag_retriever import get_embedding, get_vectors, load_embeddings, rag_answer
open_logs("image_caption")


# README: This script creates a description out of an image, and then uses that to ask a RAG pipeline if the 
# information in the knowledge pool is in accordance to the image

# can also be a path to a local image
image = "https://media.istockphoto.com/id/490734017/photo/old-factory-building-facade.jpg?s=612x612&w=0&k=20&c=Z5ixfLuF_2mNgkh5SICiPcXvpzBVvuaQqBaUe3SarqQ="
embeddings_json= "../LLM-Knowledge-Pool-RAG/knowledge_pool/Competition_brief.json"
num_results = 10


def caption_image(image: str)-> str:
    response = client.chat.completions.create(
        model=vision_model,
        messages=[
                {
                "role": "user",
                "content": [
                    {"type": "text", "text": f"""
                        Begin by generating a comprehensive portrayal of the image, focusing on architectural attributes, aesthetics, material composition, and the ambiance conveyed by the structure. 
                        Delve into specifics such as the facade, windows, and any discernible elements. 
                        """,
                    },
                    {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{get_base_64_img(image)}"
                    },
                    },
                ],
                }
            ],
            max_tokens=1000,
    )
    return response.choices[0].message.content

def use_rag(caption, embeddings_json, num_results):
    print("Initiating RAG...")
    # Embed our image caption
    question_vector = get_embedding(caption)

    # Load the knowledge embeddings
    index_lib = load_embeddings(embeddings_json)

    # Retrieve the best vectors
    scored_vectors = get_vectors(question_vector, index_lib, num_results)
    scored_contents = [vector['content'] for vector in scored_vectors]
    rag_result = "\n".join(scored_contents)

    # Get answer from rag informed agent
    system_prompt = """Answer the question based on the provided information. 
                You are given the extracted parts of a long document (provided information) and a description of an image. 
                Provide a detailed answer about wheather the image description relates to the principles given in the provided information.
                If you don't know the answer, just say "I do not know." Don't make up an answer."""
    
    question = f"""PROVIDED INFORMATION: {rag_result}
                IMAGE DESCRIPTION: {caption}"""
    
        

    answer = rag_answer(question, system_prompt)
    return answer

# Get the description of the LoRA image
caption = caption_image(image)

# Pass it to the RAG alongside the retrieved vectors using that caption
rag_result= use_rag(caption, embeddings_json, num_results)
print(rag_result)




close_logs()