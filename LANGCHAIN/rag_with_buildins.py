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
from langchain_community.document_loaders import TextLoader
from langchain.chains import create_retrieval_chain

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from key_reader import langchain_key, tavily_key, openai_key

"""
We need no function to concatenate input documents and keep track of the sources 

We define a system prompt that specifies system instructions and has a placeholder for user input

The same Chain as Build in the rag_naive can be build with build-in functions

OLD:
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
"""

def to_splits(loader):
    docs = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    return all_splits

loader = WebBaseLoader("https://www.futurebrains.io/team/nicolas-a-durr")
all_splits = to_splits(loader)
print(f"Website Chunks: {len(all_splits)}")

# Initialize an empty vector store
vectorstore = Chroma(collection_name="robins_collection", embedding_function=OpenAIEmbeddings())
vectorstore.add_documents(documents=all_splits)

loader = TextLoader("nick.txt")
all_splits = to_splits(loader)
print(f"Nick Text Chunks: {len(all_splits)}")
vectorstore.add_documents(documents=all_splits)

loader = TextLoader("robin.txt")
all_splits = to_splits(loader)
print(f"Robin Text Chunks: {len(all_splits)}")
vectorstore.add_documents(documents=all_splits)


retriever = vectorstore.as_retriever(search_kwargs={"k": 6})
llm = ChatOpenAI(model="gpt-4o-mini")

system_prompt = (
    "You are an assistant for question-answering tasks. "
    "Use the following pieces of retrieved context to answer "
    "the question. If you don't know the answer, say that you "
    "don't know. Use three sentences maximum and keep the "
    "answer concise."
    "answer in german"
    "\n\n"
    "{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
# specifies how retrieved context is fed into a promp
# stuff means no further processing like summarization
question_answer_chain = create_stuff_documents_chain(llm, prompt)

# adds the retrieval step and propagates the retrieved context through the chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)

response = rag_chain.invoke({"input": "How could Nicolas Robin and Nick work together?"})

# response["answer"]: Plain answer
print(f"Answer: {response['answer']}")

print("\nSources of the Answer:")
# response["context"]: List of Source documents
for i, document in enumerate(response["context"]):
    print(f"{i+1}. Source: {document}")

