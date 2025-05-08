import os
import json
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
pc = Pinecone(api_key=PINECONE_API_KEY)

# Index name
index_name = "cheese-knowledge"

# Get the index
index = pc.Index(index_name)

def embed_text(text):
    """Generate embedding for the given text using OpenAI."""
    response = client.embeddings.create(
        input=[text],
        model="text-embedding-3-small"
    )
    return response.data[0].embedding

def ingest():
    """Ingest cheese data into Pinecone."""
    try:
        # Load cheese data
        with open("cheese_data.json", "r", encoding="utf-8") as f:
            cheeses = json.load(f)
        
        print(f"Loaded {len(cheeses)} cheese products from cheese_data.json")
        
        # Process each cheese
        for i, cheese in enumerate(cheeses):
            # Create context string
            context = f"{cheese['description']}. Brand: {cheese['brand']}. Price: {cheese['total_price']}. Unit Price: {cheese['unit_price']}"
            
            # Get embedding
            embedding = embed_text(context)
            
            # Upload to Pinecone
            index.upsert([
                {
                    "id": str(i),
                    "values": embedding,
                    "metadata": cheese
                }
            ])
            print(f"Uploaded cheese {i+1} of {len(cheeses)}")
        
        print("Ingestion completed successfully!")
    
    except Exception as e:
        print(f"An error occurred during ingestion: {str(e)}")

if __name__ == "__main__":
    ingest()