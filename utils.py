#Import deps
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
import tiktoken
from langchain.chat_models import ChatOpenAI 
import os
from langchain.agents import Tool
from langchain.agents import load_tools
from sqlalchemy import MetaData
from sqlalchemy import create_engine
from sqlalchemy import Column, Integer, String, Table, Date, Float
import pandas as pd
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent
import streamlit as st
from dotenv import load_dotenv
from langchain.chains import RetrievalQA

# # Load variables from .env file
# load_dotenv()

# # Access the API key
# openai_api_key = os.getenv('OPENAI_API_KEY')

OPENAI_API_KEY = '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'
# Intialize the LLM text-davinci-003 model
@st.cache_resource
def load_llm_model(temperature = 0.5):
    llm = OpenAI(
        openai_api_key = OPENAI_API_KEY,
        model_name = 'text-davinci-003',
        temperature = temperature
    )
    return llm



@st.cache_resource
def load_gptturbo_model(temperature = 0.5):
    llm = OpenAI(
        openai_api_key = OPENAI_API_KEY,
        model_name = 'gpt-3.5-turbo',
        temperature = temperature
    )
    return llm


@st.cache_resource
def openai_embedding():
    embeddings = OpenAIEmbeddings(
        model='text-embedding-ada-002',
        openai_api_key=OPENAI_API_KEY
    )
    return embeddings


@st.cache_data
def tiktoken_len(text):
    tokenizer = tiktoken.get_encoding('cl100k_base')
    tokens = tokenizer.encode(
        text,
        disallowed_special=()
    )
    return len(tokens)



@st.cache_data
def split_text(chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(
                chunk_size = chunk_size,
                chunk_overlap = chunk_overlap,
                length_function=len,
                #separators=["\n\n", "\n", " ", ""]
    )
    return text_splitter


@st.cache_data
def create_memory():
    conversational_memory = ConversationBufferWindowMemory(
        memory_key='chat_history',
        k=5,
        return_messages=True
    )
    return conversational_memory

@st.cache_data
def create_retrievalqa(model, retriever):
    chromaqa = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=retriever.as_retriever() 
    )
    return chromaqa



def card():
    return """
        <div class="card" style="width: 18rem;">
            <div class="card-body">
                <h5 class="card-title">Card title</h5>
                <h6 class="card-subtitle mb-2 text-muted">Card subtitle</h6>
                <p class="card-text">Some quick example text to build on the card title and make up the bulk of the card's content.</p>
                <a href="#" class="card-link">Card link</a>
                <a href="#" class="card-link">Another link</a>
            </div>
        </div> 
    """
