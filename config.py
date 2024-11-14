from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from .env file
load_dotenv()

# Access the Pinecone API key and environment
api_key = os.getenv("PINECONE_API_KEY")
environment = "us-east-1"  # Example environment, update as per your setup

# Check if the API key is available
if api_key is None:
    raise ValueError("PINECONE_API_KEY not found. Please check your .env file")

# Initialize Pinecone
pc = Pinecone(api_key=api_key, environment=environment)

# Now check or create index as needed
if "basic-rag-project" not in pc.list_indexes():
    pc.create_index(
        name="basic-rag-project",
        dimension=384,  # Adjust to your model's dimension
        metric="cosine",  # Choose the appropriate metric
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
