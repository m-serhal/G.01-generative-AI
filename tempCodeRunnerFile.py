import nest_asyncio
import os
from llama_parse import LlamaParse
from llama_index.core import SimpleDirectoryReader

# Apply nest_asyncio to allow async operations in environments with existing event loops
nest_asyncio.apply()

# Initialize LlamaParse with the API key and desired settings
parser = LlamaParse(
    api_key="llx-M8NGNirKkUNbCooPCPY6k4MRMUB9ywHUI5RN3rXFCUKC4eFC",  # Replace with your actual LlamaParse API key or set in your environment as LLAMA_CLOUD_API_KEY
    result_type="markdown",  # "markdown" and "text" are available
    verbose=True,
)

# Specify which file types the parser should handle
file_extractor = {".pdf": parser}

# Use SimpleDirectoryReader to read all files in the specified directory and apply the parser to PDF files
documents = SimpleDirectoryReader("knowledge_pool", file_extractor=file_extractor).load_data()

# Check the output
for doc in documents:
    output_filename = os.path.splitext(os.path.basename(doc.filepath))[0]
    output_path = os.path.join("knowledge_pool", f"{output_filename}.txt")
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(doc.text)
        print(f"Finished parsing {doc.filepath} and saved to {output_path}")
