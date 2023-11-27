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



    # template = """/
    # {context}
    # Question: {question}
    # Answer: 
    # """

    # PROMPT = PromptTemplate(template=template, input_variables=['question'])


    qa = RetrievalQA.from_chain_type(
            llm=ChatOpenAI(
            model_name='gpt-3.5-turbo-16k'       
        ),
        chain_type="stuff",
        # chain_type_kwargs={"prompt": PROMPT},
        retriever=docsearch.as_retriever(),
        verbose=True,
        return_source_documents=True
    )

    # query = "What did the president say about Ketanji Brown Jackson"
    # qa.run(query)

    # qa = VectorDBQA.from_chain_type(
    #     llm=OpenAI(), chain_type="stuff", vectorstore=docsearch, return_source_documents=True
    # )



    error = "TypeError: can only concatenate str (not 'list') to str"

    query = f"When running the code base script I get the error {error}.\
        Code base is in the vector database, you can look it up there.\
        Adjust the code to resolve this issue."
    

    # result = qa({"query": query})

    result = qa(query)
    print(result)

    print(len(result))

    print(result[0])
    print(result[1])

    with open("check.json", "w") as outfile:
        json.dump(result, outfile)

    # create the agent
    # python_agent_executor = create_python_agent(
    #     llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    #     tool=PythonREPLTool(),
    #     agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    #     verbose=True,
    # )

    # python_agent_executor.run(
    #     f"You will be given the following code {result}.Test this code.\
    #     If there are no issues, save this code to the current working directory.\
    #     Name the file solution and use extension .py\
    #     If there are issues, return the issue and the description indicating the problem."
    # )



    # # tools = [
    # #     Tool(
    # #         name="Vector Database of python code base",
    # #         func=qa.run,
    # #         description="useful to study the code base for when you need to answer questions about errors in the code base and create new code. Input should be a fully formed question with the error specified.",
    # #     ),
    # #     # Tool(
    # #     #     name="Python running envionment space",
    # #     #     func=python_agent_executor.run,
    # #     #     description="useful when you need to transform natural language and write from it python\
    # #     #         test python code and return the results of the code execution\
    # #     #         DO NOT SEND PYTHON CODE TO THIS TOOL only text query",
    # #     # ),
    # # ]

    # # grand_agent = initialize_agent(
    # #         tools=tools,
    # #         llm=ChatOpenAI(temperature=0, model="gpt-3.5-turbo"),
    # #         agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    # #         verbose=True
    # #     )

    # # # print("\n".join([f"{tool.name}: {tool.description}" for tool in tools]))

    # # error = "TypeError: can only concatenate str (not 'list') to str"
    # # # give the query
    # # grand_agent.run(
    # #     f"When running the code base script I get the error {error}.\
    # #     Code base is in the vector database, you can look it up there.\
    # #     Generate and save to the current working directory adjusted code that would fix this issue.\
    # #     Name the file solution"
    # # )

    # # # # Construct the agent. We will use the default agent type here.
    # # # # See documentation for a full list of options.

    # # # agent = initialize_agent(
    # # #     tools, llm, agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True
    # # # )

