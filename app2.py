import os
from apikey import openai_api_key
import streamlit as st
from langchain import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import pandas as pd



# Set the APIs
os.environ['OPENAI_API_KEY'] = openai_api_key

st.title('Ask me anything 🤖')


input = st.text_input('Plug-in your prompt input')

# Prompt template
title_prompt = PromptTemplate(
    input_variables = ['input_text'],
    template = "Write me a YouTube video title about {input_text}"
)

## LLM
llm = OpenAI(temperature = 0.9)

# LLMChain
llm_chain = LLMChain(llm=llm, prompt=title_prompt)

if input:
    #output = llm_chain.run(input_text=input)
    output = llm(input)

    st.write(output)
    