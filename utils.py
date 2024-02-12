#Import deps
import os
import yaml
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain import hub
from langchain.agents import AgentExecutor
from langchain_experimental.tools import PythonREPLTool
from langchain.agents import create_react_agent
from langchain_experimental.pal_chain import PALChain
from langchain.agents import load_tools
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain.vectorstores import Pinecone
import pinecone
from langchain_core.runnables import (
    RunnableParallel,
    RunnablePassthrough
)
from langchain.schema.output_parser import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# Access the API key
api_key = ''

# Loading the prompts:
with open("prompts.yaml", "r") as yaml_file:
        data = yaml.safe_load(yaml_file)


# LLM Model
def load_llm_model(temperature: float = 0.0):

    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=temperature,
        google_api_key=api_key
    )
    return llm

# Python agent
def load_python_agent(model):

    python_tool_instructions = data['python_tool_instructions']
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    prompt = base_prompt.partial(instructions=python_tool_instructions)
    # Creating Python Tool
    python_tool = [PythonREPLTool()]
    # Creating a Python agent
    python_agent = create_react_agent(llm=model, tools=python_tool, prompt=prompt)
    #Agent Executor
    agent_executor = AgentExecutor(agent=python_agent, tools=python_tool, verbose=True)
    return agent_executor



def load_pal_agent(model):

    pal_chain = PALChain.from_math_prompt(llm=model, verbose=True)
    return pal_chain




def load_serp_agent(model):

    os.environ['SERPAPI_API_KEY'] = ''
    serp_tool = load_tools(tool_names=["serpapi"])
    base_prompt = hub.pull("langchain-ai/react-agent-template")
    serp_tool_instructions = data['serp_tool_instructions']
    prompt = base_prompt.partial(instructions=serp_tool_instructions)
    serp_agent = create_react_agent(
        llm=model,
        tools=serp_tool,
        prompt=prompt)
    serp_agent = AgentExecutor(agent=serp_agent, tools=serp_tool, verbose=True)
    return serp_agent



def load_vector_agent(model):
    # Pinecone Initialization
    YOUR_API_KEY = ''
    YOUR_ENV = "us-west4-gcp-free"
    text_field = "text"
    index_name = 'wwts'
    # Initialize the Pinecone client.
    pinecone.init(
        api_key=YOUR_API_KEY,
        environment=YOUR_ENV
    )
    index = pinecone.Index(index_name)
    # Instantiate the Google embedding model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    # Create Pinecone VectorDB object using with Langchain
    vectorstore = Pinecone(index=index, embedding=embeddings.embed_query, text_key=text_field)
    retriever = vectorstore.as_retriever()
    vector_prompt_template = data['vector_prompt_template']
    prompt = ChatPromptTemplate.from_template(vector_prompt_template)
    # Build RAG
    rag = RunnableParallel(
        {"context": retriever, "question": RunnablePassthrough()}
    )
    # Output parser to convert output to string
    output_parser = StrOutputParser()
    # Building RAG chain
    rag_agent = rag | prompt | model | output_parser

    return rag_agent

#
#
# def card():
#     return """
#         <div class="card" style="width: 18rem;">
#             <div class="card-body">
#                 <h5 class="card-title">Card title</h5>
#                 <h6 class="card-subtitle mb-2 text-muted">Card subtitle</h6>
#                 <p class="card-text">Some quick example text to build on the card title and make up the bulk of the card's content.</p>
#                 <a href="#" class="card-link">Card link</a>
#                 <a href="#" class="card-link">Another link</a>
#             </div>
#         </div>
#     """
