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
# Load CSS
def load_css(css_file):
    with open(css_file, "r") as f:
        css = f.read()
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

# Load the CSS
css_path = os.path.join(os.path.dirname(__file__), "style.css")
load_css(css_path)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []

if "previous_answer" not in st.session_state:  # Add this initialization
    st.session_state.previous_answer = ""
# os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'

# Sidebar for API Keys and Info
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/cheese.png", width=100)
    st.title("""🧀 Cheese Bot\n\n""")
    
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
    if  st.button("🗑️ Clear Chat History", 
                     help="Click to clear all chat history",
                     type="secondary",
                     use_container_width=True):
            st.session_state.messages = []
            st.session_state.previous_answer = ""
            st.rerun()
    # if  st.button("🗑️ Save Chat History", 
    #                  help="Click to save all chat history",
    #                  type="primary",
    #                  use_container_width=True):
    #     with open("chat_history.txt", "w", encoding="utf-8") as f:
    #         f.write(st.session_state.messages)
    
# Main content area
st.title("          🧀 Cheese Explorer")
st.markdown("       Ask me anything about cheeses from shop.kimelo.com!")
st.markdown("--------------------------------")
# Chat container with custom styling
chat_container = st.container()

# Display chat history in a beautiful way
with chat_container:
    for message in st.session_state.messages:
        if message["role"] == "user":
            with st.chat_message("user", avatar="🧑"):
                st.markdown(f"<div class='user-message'>{message['content']}</div>", unsafe_allow_html=True)
        else:
            with st.chat_message("assistant", avatar="🧀"):
                st.markdown(f"<div class='assistant-message'>{message['content']}</div>", unsafe_allow_html=True)

# Initialize OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
assert OPENAI_API_KEY is not None, "OPENAI_API_KEY environment variable not set"
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
try:
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
    assert PINECONE_API_KEY is not None, "PINECONE_API_KEY environment variable not set"
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "cheese-knowledge"
    try:
        index = pc.Index(index_name)
        # Verify index connection with a simple query
        index.describe_index_stats()
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

def search_pinecone(query, top_k=5):
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
            # print(f"Using filter: {filter_dict}")
            
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

def ask_gpt(question, context,previous_answer):
    try:
        prompt = f"""
            When a user asks a question, do the following:

            - If it's about cheese, answer using the cheese data only.
                    You are an expert cheese sommelier and product specialist. Answer the user's question using the provided cheese information in comprehensive detail, including product details and shopping information.
                    
                    CHEESE INFORMATION:
                    {context}
                    you must provide the information that user asked for.
                    only if user demand more information , you must answer the question including:
                        
                        1. PRODUCT INFORMATION:
                        - Product name and brand
                        - URL where the product can be purchased (format as clickable link)
                        - SKU/UPC codes for reference
                        - Include image URLs in your response (format as markdown: ![Cheese Image](image_url)) 
                        - Price information, weights, and packaging options
                        
                        2. CHEESE CHARACTERISTICS:
                        - FLAVOR PROFILE: Describe the complex flavors, aromas, taste progression, and intensity
                        - TEXTURE: Detail the mouthfeel, consistency, and physical characteristics  
                        - APPEARANCE: Describe the color, rind, interior, and visual aspects
                        - ORIGIN: Explain the geographical and cultural significance of this cheese
                        
                Guidelines:
                • Format your response in a clean, organized way with clear sections and markdown formatting
                • Include ALL available product details (URLs, SKUs, images, pricing)
                • If showing multiple products, create a separate section for each with its own details
                • For images, include at least one image URL formatted as markdown if available
                • Include links to the product and related/similar products formatted as markdown
                • Be thorough but conversational, like an enthusiastic cheese expert sharing their passion
            - If it's a general food-related question (not about cheese), give a common, non-political, non-character-based answer and generate image.
            - Use casual American English.
            
            Answer this answer is previous your answer:{previous_answer}
            User ask the question related to previous answer, You must answer the question based on previous answer and user question.
            User question: {question}
            """
        # print(f"Prompt: {prompt}\n\n")
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
st.markdown("""
    <style>
        div[data-testid="chat-input"] textarea {
            width: 300px !important;
            min-width: 300px !important;
            max-width: 300px !important;
        }
    </style>
""", unsafe_allow_html=True)
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
                            price: {cheese['price']}. Cost per pound: {cheese['Cost per pound']}. Unit: {cheese['Unit']}. Weight: {cheese['weight(pound)']}. standard: {cheese['standard']}. image_url: {cheese['image_url']}"
                        for cheese in contexts
                    ])
                    
                    answer = ask_gpt(prompt, context_text , st.session_state.previous_answer)
                    st.session_state.previous_answer = answer
                    # Display answer with custom styling\                    
                    st.markdown(f"<div class='assistant-message'>{answer}</div>", unsafe_allow_html=True)
                    
                    # Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer
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

