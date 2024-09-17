from langchain_openai import ChatOpenAI
import bs4
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.document_loaders import WebBaseLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain import hub

from key_reader import langchain_key, tavily_key, openai_key

"""
Retrieval Augmented Generation

    Learned: Embedd data, retrieve chunks

    Main Components: LOAD, SPLIT, EMBEDD, STORE, RETRIEVE, GENERATE
     
     - Indexing: a pipeline for ingesting data from a source and indexing it. -> EMBEDD, STORE
     - Retrieval & Generation: the actual RAG chain, which takes the user query at run time and retrieves the relevant data

Chunk Metadata lets us find its Source

Use the langchain prompt hub for templates, e.g. for instructions like 
"""

# bs4: Beautiful Soup is a web scraping tool
# Soupstrainer filter HTML elements based on criteria

# Load, chunk and index the contents of the blog.
loader = WebBaseLoader(
    web_paths=("https://lilianweng.github.io/posts/2023-06-23-agent/",),
    bs_kwargs=dict(
        parse_only=bs4.SoupStrainer(
            class_=("post-content", "post-title", "post-header")
        )
    ),
)
docs = loader.load()

# overlaps in the chunks: preserve context, improve relevance of retrieved chunk
# add_start_index: adds metadata 'start_index': index of chunk-start
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=200, add_start_index=True
)
all_splits = text_splitter.split_documents(docs)
len(all_splits)

# Chroma -> Vector Database
# Embedd the contents of each document
# At this point we have a query-able vector store
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# Retrieve and generate using the relevant snippets of the blog.
# Search k nearest embeddings
retriever = vectorstore.as_retriever(search_kwargs={"k": 1})
prompt = hub.pull("rlm/rag-prompt")
# There are many retriever techniques:
# - MultiQueryRetriever generates variants of the input question
# - MultiVectorRetriever instead generates variants of the embeddings
# - Use Metadata Filters

llm = ChatOpenAI(model="gpt-4o-mini")


# This is a prompt template for a Q&A chatbot with system instructions
"""
placeholders need to be filled, see invoke
Question: {question} 
Context: {context} 
"""
prompt = hub.pull("rlm/rag-prompt")

example_messages = prompt.invoke(
    {"context": "Job Interview", "question": "Can you help me with my Coding Challenge"}
).to_messages()

example_messages


# Iterate over all documents and get content
# Concatenate all documents
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# How can the source documents be tracked?
# -> create_retrieval_chain method


# The real values filling the placeholders are created (dict). RunnablePassthrough() passes Question unchanged
# The Template stored in prompt is filled with the values
# It is passed to the LLM
# Response is parsed
rag_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | prompt
    | llm
    | StrOutputParser()
)

rag_chain.invoke("What is Task Decomposition?")

for chunk in rag_chain.stream("What is Task Decomposition?"):
    print(chunk, end="", flush=True)


