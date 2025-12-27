# create_index.py - Run this ONLY ONCE

from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
import time

load_dotenv()

api_key = os.environ.get("PINECONE_API_KEY")
if not api_key:
    raise ValueError("PINECONE_API_KEY not found in .env")

pc = Pinecone(api_key=api_key)

index_name = "pdfchatbot"

# List existing indexes
existing_indexes = [idx.name for idx in pc.list_indexes()]

if index_name in existing_indexes:
    print(f"Index '{index_name}' already exists. You're good to go!")
else:
    print(f"Creating new index: {index_name}")
    pc.create_index(
        name=index_name,
        dimension=384,  # Matches sentence-transformers/all-MiniLM-L6-v2
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )

    # Wait until ready
    while not pc.describe_index(index_name).status["ready"]:
        print("Waiting for index to be ready...")
        time.sleep(10)

    print(f"Index '{index_name}' created and ready! 🎉")

print("You can now run your PDF Chatbot.")