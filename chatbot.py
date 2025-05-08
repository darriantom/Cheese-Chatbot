import os
import json
import streamlit as st
import ssl
import certifi
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
import urllib3

# Disable SSL warnings
urllib3.disable_warnings()

# Configure SSL context
ssl_context = ssl.create_default_context(cafile=certifi.where())

# API Keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI
client = OpenAI(api_key=OPENAI_API_KEY)

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("cheese-knowledge")
    st.success("Successfully connected to Pinecone!")
except Exception as e:
    st.error(f"Error connecting to Pinecone: {str(e)}")
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

def search_pinecone(query, top_k=3):
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
        prompt = f"Answer the question based on the following cheese data:\n{context}\n\nQuestion: {question}\nAnswer:"
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.choices[0].message.content
    except Exception as e:
        st.error(f"Error getting GPT response: {str(e)}")
        return "Sorry, I encountered an error while processing your question."

# Streamlit UI
st.title("🧀 Cheese RAG Chatbot")
st.write("Ask anything about cheeses from shop.kimelo.com!")

user_question = st.text_input("Your question about cheese:")

if user_question:
    with st.spinner("Fetching answer..."):
        try:
            contexts = search_pinecone(user_question)
            if contexts:
                context_text = "\n\n".join([
                    f"{c['description']}: Brand: {c['brand']} (Price: {c['total_price']}, Unit Price: {c['unit_price']})" 
                    for c in contexts
                ])
                answer = ask_gpt(user_question, context_text)
                st.markdown("### Answer")
                st.write(answer)
                st.markdown("### Reference Data")
                st.info(context_text)
            else:
                st.warning("No relevant cheese information found. Please try a different question.")
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")