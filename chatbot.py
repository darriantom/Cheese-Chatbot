import os
import json
import streamlit as st
import ssl
import certifi
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import urllib3

from langchain.chat_models import init_chat_model
from langchain_core.prompts import ChatPromptTemplate

from transformers import pipeline
# Disable SSL warnings
urllib3.disable_warnings()

# Configure SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())
os.environ['TRANSFORMERS_CACHE'] = '/tmp/huggingface_cache'
# Streamlit UI
st.title("🧀 Cheese RAG Chatbot")
st.write("Ask anything about cheeses from shop.kimelo.com!")

# Initialize API keys
if 'openai_api_key' not in st.session_state:
    st.session_state.openai_api_key = ""
if 'pinecone_api_key' not in st.session_state:
    st.session_state.pinecone_api_key = ""

# API Key Input Section
with st.expander("API Keys Configuration"):
    st.session_state.openai_api_key = st.text_input("Your OpenAI API Key:", type="password", value=st.session_state.openai_api_key)
    st.session_state.pinecone_api_key = st.text_input("Your Pinecone API Key:", type="password", value=st.session_state.pinecone_api_key)

# Initialize OpenAI
client = OpenAI(api_key=st.session_state.openai_api_key)

# Initialize Pinecone
try:
    if st.session_state.pinecone_api_key:
        pc = Pinecone(api_key=st.session_state.pinecone_api_key)
        index = pc.Index("cheese-knowledge")
        st.success("Successfully connected to Pinecone!")
    else:
        st.warning("Please enter your Pinecone API key to continue.")
except Exception as e:
    st.error(f"Error connecting to Pinecone: {str(e)}")
    st.stop()

def embed_text(text):
    try:
        if not st.session_state.openai_api_key:
            st.error("Please enter your OpenAI API key to continue.")
            return None
        response = client.embeddings.create(
            input=[text],
            model="text-embedding-3-small"
        )
        return response.data[0].embedding
    except Exception as e:
        st.error(f"Error generating embedding: {str(e)}")
        return None

def search_pinecone(query, top_k=24):
    try:
        embedding = embed_text(query)
        if embedding is None:
            return []
        
        results = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )
        return [match['metadata'] for match in results['matches']]
    except Exception as e:
        st.error(f"Error searching Pinecone: {str(e)}")
        return []

def ask_gpt(question, context):
    try:
        prompt = f"here is some data. {context}> Now when you come across user query , you first decide whether \
            user wants common answer or, answer based on our cheese information data. If you thought user wants the information based on our cheese information data,\
                you must answer only based on out cheese information data. And If you thought user wants the common answer, you must answer only in common way.\
                    you must answer in style of American english so that users understand easily. you must answer in short and concise way. Question: {question}\nAnswer:"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting GPT response: {str(e)}")
        return "Sorry, I encountered an error while processing your question."

# User Question Input
user_question = st.text_input("Your question about cheese:")

if user_question:
    if not st.session_state.openai_api_key or not st.session_state.pinecone_api_key:
        st.error("Please enter both API keys to continue.")
    else:
        with st.spinner("Fetching answer..."):
            try:
                
                contexts = search_pinecone(user_question)
                if contexts:
                    context_text = "\n\n".join([
                        f"{c['description']}: Brand: {c['brand']} (Price: {c['total_price']}, Unit Price: {c['unit_price']})" 
                        for c in contexts
                    ])
                    
                    model = init_chat_model("chatgpt-4o-latest", model_provider="openai")
                    system_template = "Explain the following text: {text}"

                    prompt_template = ChatPromptTemplate.from_messages(
                        [("system", system_template), ("user", "{user_text}")]
                    )
                    
                    prompt = prompt_template.invoke({"text": "context_text", "user_text": {context_text}})
                    prompt.to_messages()
                    response = model.invoke(prompt)
                    print(f"Prompt: {response.content}")
                    
                    answer = ask_gpt(user_question, response.content)
                    st.markdown("### Answer")
                    st.write(answer)
                    st.markdown("### Reference Data")
                    st.info(context_text)
                    st.info(response.content)
                else:
                    st.warning("No relevant cheese information found. Please try a different question.")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")