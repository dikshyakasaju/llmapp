#Import deps
import os
from utils import *
import streamlit as st
from streamlit_extras.add_vertical_space import add_vertical_space


# Initialize llm model
llm = load_llm_model()

st.set_page_config(layout="wide")
#Page Title
st.title('Ask me anything 🤖')


# Set up Streamlit Sidebar contents
with st.sidebar:
    st.info('This app utilizes various LLMs to streamline tasks efficiently. Choose your preferred LLM and Agent below for seamless productivity!')
    # model = st.radio('Choose your LLM model', ['text-davinci-003', 'gpt-3.5-turbo'])
    agent = st.radio('Choose your Agent', ['General', 'Python', 'Math', 'SERP', 'Vectorstore'])


    add_vertical_space(2)
    st.write('Made by Dikshya')


if agent =='Python': 
    st.info('Leverage Python Agent to execute python commands.')
    input = st.text_input("Plug-in your input")
    # Load Python Agent
    python_agent = load_python_agent(model=llm)

    if input:
        with st.spinner('Processing...'):
             # Pass the prompt to the Agent
            response = python_agent.invoke({"input": input})['output']
            st.write(response)



if agent =='Math':
    st.info('Leverage Math Agent for hard word math problem.')
    input = st.text_input("Plug-in your input")
    #  Load PAL Agent
    pal_agent = load_pal_agent(model=llm)
    if input:
         # Pass the prompt to the Agent
        with st.spinner('Processing...'):
            response = pal_agent.invoke(input)
            st.write(response)



if agent =='SERP':
    st.info('Leverage SERP Agent to answer questions about current events.')
    input = st.text_input("Plug-in your input")
    # Load SERP Agent
    serp_agent = load_serp_agent(model=llm)
    if input:
        with st.spinner('Processing...'):
            # Pass the prompt to the Agent
            response = serp_agent.invoke({"input": input})['output']
            st.write(response)



if agent =='Vectorstore':
    st.info('Leverage Vectorstore Agent to query Source Knowledge.')
    input = st.text_input("Plug-in your input")
    # Load VectorDB Agent
    vector_agent = load_vector_agent()
    if input:
        # Pass the prompt to the Agent
        with st.spinner('Processing...'):
            response = vector_agent.invoke(input)
            st.write(response)




if agent =='General':
    st.info('Leverage General Agent to ask any generic questions')
    input = st.text_input("Plug-in your input")
    if input:
        with st.spinner('Processing...'):
             # Pass the prompt to the Agent
            response = llm.invoke(input).content
            st.write(response)



#     # st.write(output)
#     # st.write(docs)
#     # st.write(text_chunks)
#     # st.balloons()
