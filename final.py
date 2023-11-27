import os
from dotenv import load_dotenv
from langchain.document_loaders import TextLoader, PythonLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Pinecone
# from langchain import VectorDBQA, OpenAI
from langchain.chains import VectorDBQA
from langchain.llms import OpenAI
import pinecone
from langchain.agents import AgentType, Tool, initialize_agent
from langchain.chat_models import ChatOpenAI
from langchain_experimental.tools import PythonREPLTool
from langchain_experimental.agents.agent_toolkits import create_python_agent
from langchain.chains import RetrievalQA
from langchain import PromptTemplate
import json
load_dotenv()

pinecone.init(
    api_key=os.environ.get("PINECONE_API_KEY"),
    environment=os.environ.get("PINECONE_ENV"),
)

index_name = "ai-developer"
index_dim = 1536
index_metric = "euclidean"

if index_name in pinecone.list_indexes():
    pass
else:
    pinecone.create_index(index=index_name, dimension=index_dim, metric=index_metric)

if __name__ == "__main__":
    
    # load the file
    loader = PythonLoader(
        "load_py_file.py"
    )
    document = loader.load()

    # split the file
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(len(texts))

    # create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=os.environ.get("OPENAI_API_KEY"))
    docsearch = Pinecone.from_documents(
        texts, embeddings, index_name=index_name
    )