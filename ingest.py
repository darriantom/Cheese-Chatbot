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

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("Missing required API keys. Please check your .env file.")

# Initialize OpenAI
try:
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception as e:
    raise Exception(f"Failed to initialize OpenAI client: {str(e)}")

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
except Exception as e:
    raise Exception(f"Failed to initialize Pinecone client: {str(e)}")

# Index name and configuration
index_name = "cheese-knowledge"
dimension = 1536  # Dimension for text-embedding-3-small

# Get or create the index
try:
    # Check if index exists
    existing_indexes = pc.list_indexes()
    if index_name in [index.name for index in existing_indexes]:
        # Delete existing index
        pc.delete_index(index_name)
        print(f"Deleted existing index: {index_name}")
    
    # Create new index with correct dimension
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric="cosine",
        spec=dict(
            serverless=dict(
                cloud="aws",
                region="us-east-1"
            )
        )
    )
    print(f"Created new index: {index_name} with dimension {dimension}")
    
    # Get the index
    index = pc.Index(index_name)
except Exception as e:
    raise Exception(f"Failed to setup Pinecone index: {str(e)}")

def embed_text(text):
    """Generate embedding for the given text using OpenAI."""
    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"  # Using 3-small which has 1536 dimensions
        )
        return response.data[0].embedding
    except Exception as e:
        raise Exception(f"Failed to generate embedding: {str(e)}")

def ingest():
    """Ingest cheese data into Pinecone."""
    try:
        # Load cheese data
        try:
            with open("cheese_data.json", "r", encoding="utf-8") as f:
                cheeses = json.load(f)
        except FileNotFoundError:
            raise Exception("cheese_data.json file not found")
        except json.JSONDecodeError:
            raise Exception("Invalid JSON format in cheese_data.json")
        
        print(f"Loaded {len(cheeses)} cheese products from cheese_data.json")
        
        # Process each cheese
        for i, cheese in enumerate(cheeses):
            try:
                # Validate required fields
                required_fields = ['product_name', 'company_name', 'price', 'Unit', 
                                 'Cost per pound', 'standard', 'weight(pound)', 'SKU', 'UPC']
                for field in required_fields:
                    if field not in cheese:
                        raise ValueError(f"Missing required field '{field}' in cheese data at index {i}")
                
                # Create context string
                context = f"product_name: {cheese['product_name']}. company_name: {cheese['company_name']}. \
                    Price: {cheese['price']}. Unit: {cheese['Unit']}. Cost per pound: {cheese['Cost per pound']}. \
                        Standard: {cheese['standard']}. Weight: {cheese['weight(pound)']}. SKU: {cheese['SKU']}. UPC: {cheese['UPC']}. image_url: {cheese['image_url']}"
                
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
                
            except Exception as cheese_error:
                print(f"Error processing cheese at index {i}: {str(cheese_error)}")
                continue
        
        print("Ingestion completed successfully!")
    
    except Exception as e:
        print(f"An error occurred during ingestion: {str(e)}")
        raise

if __name__ == "__main__":
    try:
        ingest()
    except Exception as e:
        print(f"Fatal error: {str(e)}")
        exit(1)