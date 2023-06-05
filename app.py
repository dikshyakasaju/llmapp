#Import deps
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA
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
from langchain.agents import AgentType
import streamlit as st
from utils import load_llm_model, load_gptturbo_model, openai_embedding, split_text
from streamlit_extras.add_vertical_space import add_vertical_space
from langchain.agents.agent_toolkits import create_python_agent
from langchain.tools.python.tool import PythonREPLTool
from dotenv import load_dotenv

# Load variables from .env file
# load_dotenv()

# openai_api_key = os.environ['OPENAI_API_KEY']
OPENAI_API_KEY = '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'

st.set_page_config(layout="wide")
# Page Title
st.title('Ask me anything 🤖')


# Set up Streamlit Sidebar contents
with st.sidebar:
    st.info('This app utilizes various LLMs to streamline tasks efficiently. Choose your preferred LLM and Agent below for seamless productivity!')
    model = st.radio('Choose your LLM model', ['text-davinci-003', 'gpt-3.5-turbo'])
    agent = st.radio('Choose your Agent', ['General', 'Python', 'Math', 'SERP', 'Vectorstore'])

    
    if model == 'gpt-3.5-turbo': 
        # Instance of gpt-3.5-turbo model
        llm = load_gptturbo_model(temperature=0.5) 
    else: 
        # Instance of text-da-vinci model
        llm = load_llm_model(temperature=0.5)
    
    add_vertical_space(2)
    st.write('Made by Dikshya')


if agent =='Python': 
    st.info('Leverage Python Agent to execute python commands.')
    input1 = st.text_input("Plug-in your input")
    # Python agent
    python_agent = create_python_agent(llm=llm, tool=PythonREPLTool(), verbose=True)
    if input1:
        with st.spinner('Processing...'):
             # Pass the prompt to the Agent
            response = python_agent.run(input1) 
            st.write(response)



if agent =='Math': 
    st.info('Leverage Math Agent for hard word math problem.')
    input2 = st.text_input("Plug-in your input")
    # Math Agent
    math_tool = load_tools(tool_names=['pal-math'], llm=llm)
    math_agent = initialize_agent(
                agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                tools=math_tool,
                llm=llm,
                verbose=True
    )
    if input2:
         # Pass the prompt to the Agent
        with st.spinner('Processing...'):
            response = math_agent.run(input2) 
            st.write(response)



if agent =='SERP': 
    st.info('Leverage SERP Agent to answer questions about current events.')
    input3 = st.text_input("Plug-in your input")
    # SERP Agent
    os.environ['SERPAPI_API_KEY'] = '690b9390c6acd3c23dc2ce30f9dc88d58f84445bdb3b696a746e0b8212491b4c'
    serp_tool = load_tools(tool_names=['serpapi'], llm=llm)
    serp_agent = initialize_agent(
                agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                tools=serp_tool,
                llm=llm,
                verbose=True,
    )
    
    if input3:
        # Pass the prompt to the LLM Chain
        with st.spinner('Processing...'):
            response = serp_agent.run(input3) 
            st.write(response)



if agent =='Vectorstore': 
    st.info('Leverage Vectorstore Agent to query Source Knowledge.')
    input4 = st.text_input("Plug-in your input")
    #Load the source knowledge
    loader = TextLoader('new.txt')
    docs = loader.load()
    # Instantiating Text Splitter
    text_splitter = split_text(chunk_size=200, chunk_overlap=40)
    # Split the text
    doc_chunks = text_splitter.split_documents(docs)
    # Instantite the OpenAIEmbeddings
    embedding = openai_embedding()
    # Create a Chroma vectorstore from a list of documents
    vectorstore = Chroma.from_documents(documents=doc_chunks, embedding=embedding)
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever() 
    )
    # Create VectorStore tool
    vectordb_tool = Tool(
            name = 'VectorDB Tool',
            description='Useful when you need to query the VectorDB for source knowledge',
            func=qa.run
            )
    # Create VectorDB Agent
    vectordb_agent = initialize_agent(
                agent= AgentType.ZERO_SHOT_REACT_DESCRIPTION,
                tools=[vectordb_tool],
                llm=llm,
                verbose=True
            )
    
    if input4:
        # Pass the prompt to the Agent
        with st.spinner('Processing...'):
            response = vectordb_agent.run(input4) 
            st.write(response)




if agent =='General': 
    st.info('Leverage General Agent to ask any General questions')
    input5 = st.text_input("Plug-in your input")
    # General Agent without any pre-defined tools

    if input5:
        with st.spinner('Processing...'):
             # Pass the prompt to the Agent
            response = llm(input5) 
            st.write(response)



#     # st.write(output)
#     # st.write(docs)
#     # st.write(text_chunks)
#     # st.balloons()
