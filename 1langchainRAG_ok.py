# https://python.langchain.com/v0.1/docs/use_cases/question_answering/streaming/

import os

from langchain_openai import AzureChatOpenAI
import openai
from langchain_core.messages import HumanMessage
from azure.identity import DefaultAzureCredential, get_bearer_token_provider

#działa

from dotenv import load_dotenv

load_dotenv()



OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT") 
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT_EMBED")

# Use Azure Default Credential for authentication
credential = DefaultAzureCredential()
token_provider = get_bearer_token_provider(credential, "https://cognitiveservices.azure.com/.default")


"""
llm = AzureChatOpenAI(
    openai_api_version = OPENAI_API_VERSION,
    azure_deployment = "chat2",
    azure_endpoint = AZURE_OPENAI_ENDPOINT,
    azure_ad_token_provider = token_provider)

message = HumanMessage(content = "hi there")

print(llm.invoke([message]))

"""

################### https://python.langchain.com/v0.1/docs/use_cases/question_answering/streaming/

import bs4
from langchainhub import Client as HubClient
from langchain_community.document_loaders import WebBaseLoader
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain_text_splitters import RecursiveCharacterTextSplitter

hub = HubClient()




# Load, chunk and index the contents of the blog.

bs_strainer = bs4.SoupStrainer(class_=("post-content", "post-title", "post-header"))
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs={"parse_only": bs_strainer},
)
docs = loader.load()




text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
splits = text_splitter.split_documents(docs)
vectorstore = Chroma.from_documents(documents=splits, embedding=AzureOpenAIEmbeddings(model="text-embedding-ada-002",
                                                                                      azure_deployment = AZURE_DEPLOYMENT,
                                                                                      openai_api_version = OPENAI_API_VERSION,
                                                                                      azure_endpoint = AZURE_OPENAI_ENDPOINT,
                                                                                      azure_ad_token_provider = token_provider
                                                                                      ))

# Retrieve and generate using the relevant snippets of the blog.
retriever = vectorstore.as_retriever()
prompt = hub.pull("rlm/rag-prompt")


llm = AzureChatOpenAI(
    openai_api_version = OPENAI_API_VERSION,
    azure_deployment = "chat2",
    azure_endpoint = AZURE_OPENAI_ENDPOINT,
    azure_ad_token_provider = token_provider,
    temperature = 0)





def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


rag_chain_from_docs = (
    RunnablePassthrough.assign(context=(lambda x: format_docs(x["context"])))
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain_with_source = RunnableParallel(
    {"context": retriever, "question": RunnablePassthrough()}
).assign(answer=rag_chain_from_docs)

"""
for chunk in rag_chain_with_source.stream("What is Task Decomposition"):
    print(chunk)
"""

output = {}
curr_key = None
for chunk in rag_chain_with_source.stream("What is Task Decomposition"):
    for key in chunk:
        if key not in output:
            output[key] = chunk[key]
        else:
            output[key] += chunk[key]
        if key != curr_key:
            print(f"\n\n{key}: {chunk[key]}", end="", flush=True)
        else:
            print(chunk[key], end="", flush=True)
        curr_key = key
output