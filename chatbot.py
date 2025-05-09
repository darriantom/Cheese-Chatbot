import os
import json
import streamlit as st
from pinecone import Pinecone
from openai import OpenAI
from dotenv import load_dotenv
import urllib3
# Disable SSL warnings
urllib3.disable_warnings()

load_dotenv()
# Initialize Streamlit page config with custom theme
st.set_page_config(
    page_title="Cheese RAG Chatbot",
    page_icon="🧀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
        padding-bottom: 0px;  /* Add padding for fixed input */
    }
    .stButton button {
        border-radius: 20px;
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        flex-direction: column;
    }
    .chat-message.user {
        background-color: #f0f2f6;
    }
    .chat-message.assistant {
        background-color: #e8f4f8;
    }
    .stMarkdown {
        font-size: 1.1rem;
    }
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    /* Fixed chat input styling */
    .fixed-input {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background: white;
        padding: 1rem;
        box-shadow: 0 -2px 10px rgba(0, 0, 0, 0.1);
        z-index: 1000;
    }
    /* Adjust main content to prevent overlap */
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

previous_answer = ""
# os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'

# Sidebar for API Keys and Info
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/cheese.png", width=100)
    st.title("🧀 Cheese Bot")
    st.markdown("---")

    st.markdown("---")
    st.markdown("### 📚 About")
    st.markdown("""
        This chatbot helps you discover and learn about different types of cheeses.
        Ask questions about:
        - Cheese varieties
        - Prices and availability
        - Company information
        - Product details
    """)

# Main content area
st.title("🧀 Cheese Explorer")
st.markdown("Ask me anything about cheeses from shop.kimelo.com!")

# Chat container with custom styling
chat_container = st.container()

# Display chat history in a beautiful way
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(message["content"])
        else:
            with st.chat_message("assistant", avatar="🧀"):
                st.markdown(message["content"])
                if "images" in message and message["images"]:
                    # Create a dynamic grid for images
                    num_images = len(message["images"])
                    cols = st.columns(min(3, num_images))
                    for idx, img_path in enumerate(message["images"]):
                        if img_path:
                            col_idx = idx % 3
                            with cols[col_idx]:
                                try:
                                    st.image(
                                        img_path,
                                        use_container_width=True,
                                        caption=message.get("product_name", "")
                                    )
                                except:
                                    pass

# Initialize OpenAI

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Initialize Pinecone
try:
    pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
    index_name = "cheese-knowledge"
    try:
        index = pc.Index(index_name)
        # Verify index connection with a simple query
        index.describe_index_stats()
        st.sidebar.success("✅ Connected to Pinecone!")
    except Exception as index_error:
        st.sidebar.error(f"❌ Pinecone Error: {str(index_error)}")
        st.stop()
except Exception as e:
    st.sidebar.error(f"❌ Connection Error: {str(e)}")
    st.stop()

def embed_text(text):
    try:
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None

def get_filter_from_llm(query):
    """Get filter from LLM in a simple way"""
    try:
        prompt = f"""
            Extract a metadata filter from the user query. Return only valid JSON only using fields from this list:
            ["price", "company_name", "Unit", "Cost per pound", "standard", "weight(pound)", "image_path"]

            Examples:
            - "Show me cheeses under $20" → {{"price": {{"$lt": 20}}}}
            - "Cheeses by Tillamook" → {{"company_name": {{"$eq": "Tillamook"}}}}
            - "Show me cheeses under 5 pounds" → {{"weight(pound)": {{"$lt": 5}}}}

            Query: {query}
            """
        
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}],
            temperature=0
        )
        
        filter_str = response.choices[0].message.content.strip()
        # Clean up the response to ensure it's valid JSON
        filter_str = filter_str.replace("→", "->").strip()
        if filter_str.startswith("->"):
            filter_str = filter_str[2:].strip()
        
        try:
            return json.loads(filter_str)
        except json.JSONDecodeError:
            print(f"Failed to parse filter: {filter_str}")
            return None
            
    except Exception as e:
        print(f"Error getting filter from LLM: {str(e)}")
        return None

def search_pinecone(query, top_k=10):
    try:
        # Verify index is available
        if not index:
            st.error("Pinecone index is not initialized")
            return []
            
        embedding = embed_text(query)
        if embedding is None:
            return []
            
        # Get filter from LLM
        filter_dict = get_filter_from_llm(query)
        
        # Prepare query parameters
        query_params = {
            "vector": embedding,
            "top_k": top_k,
            "include_metadata": True
        }
        
        # Add filter if available and valid
        if filter_dict and isinstance(filter_dict, dict):
            query_params["filter"] = filter_dict
            print(f"Using filter: {filter_dict}")
            
        try:
            # Execute query
            results = index.query(**query_params)
            if not results or 'matches' not in results:
                print("No results found in Pinecone query")
                return []
            return [match['metadata'] for match in results['matches']]
        except Exception as query_error:
            st.error(f"Error executing Pinecone query: {str(query_error)}")
            return []
        
    except Exception as e:
        st.error(f"Error in search_pinecone: {str(e)}")
        return []

def ask_gpt(question, context , previous_answer):
    try:
        prompt = f"""
            You have access to cheese data: <{context}>.
            When a user asks a question, do the following:

            - If it's about cheese, answer using the cheese data only.
            - If it's a general food-related question (not about cheese), give a common, non-political, non-character-based answer and generate image.
            - Use casual American English.
            - If the question follows up on a previous answer, consider this: <{previous_answer}>.

            User Question: "{question}"
            Answer:
            """
        print(f"Prompt: {prompt}\n\n")
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting GPT response: {str(e)}")
        return "Sorry, I encountered an error while processing your question."

# Chat input at the bottom with fixed position
# st.markdown('<div class="fixed-input">', unsafe_allow_html=True)
prompt = st.chat_input("Ask a question about cheese...", key="chat_input")
st.markdown('</div>', unsafe_allow_html=True)

if prompt:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("🧀 Thinking..."):
            try:
                contexts = search_pinecone(prompt)
                if contexts:
                    context_text = "\n".join([
                        f"product_name: {cheese['product_name']}. company_name: {cheese['company_name']}. SKU: {cheese['SKU']}. UPC: {cheese['UPC']}. \
                            price: {cheese['price']}. Cost per pound: {cheese['Cost per pound']}. Unit: {cheese['Unit']}. Weight: {cheese['weight(pound)']}. standard: {cheese['standard']}"
                        for cheese in contexts
                    ])
                    
                    answer = ask_gpt(prompt, context_text, previous_answer)
                    previous_answer = previous_answer + answer
                    
                    # Display answer with custom styling
                    st.markdown(answer)
                    
                    # Display cheese images in a dynamic grid
                    cols = st.columns(min(3, len(contexts)))
                    for idx, cheese in enumerate(contexts):
                        col_idx = idx % 3
                        with cols[col_idx]:
                            if 'image_path' in cheese and cheese['image_path']:
                                try:
                                    st.image(
                                        cheese['image_path'],
                                        caption=cheese['product_name'],
                                        use_container_width=True
                                    )
                                except Exception as img_error:
                                    st.error(f"Could not load image for {cheese['product_name']}")
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "images": [cheese.get('image_path') for cheese in contexts if 'image_path' in cheese]
                    })
                else:
                    st.warning("No relevant cheese information found. Please try a different question.")
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": "No relevant cheese information found. Please try a different question."
                    })
            except Exception as e:
                error_message = f"An error occurred: {str(e)}"
                st.error(error_message)
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": error_message
                })

# Clear chat button with improved styling
st.markdown("---")
col1, col2, col3 = st.columns([1,2,1])
with col2:
    if "show_clear_confirmation" not in st.session_state:
        st.session_state.show_clear_confirmation = False
        
    if not st.session_state.show_clear_confirmation:
        if st.button("🗑️ Clear Chat History", 
                     help="Click to clear all chat history",
                     type="secondary",
                     use_container_width=True):
            st.session_state.show_clear_confirmation = True
            st.rerun()
    else:
        st.warning("Are you sure you want to clear the chat history?")
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button("Yes, Clear All", type="primary", use_container_width=True):
                st.session_state.messages = []
                st.session_state.show_clear_confirmation = False
                st.rerun()
        with col_no:
            if st.button("No, Keep History", type="secondary", use_container_width=True):
                st.session_state.show_clear_confirmation = False
                st.rerun()