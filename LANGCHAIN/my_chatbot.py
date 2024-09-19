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
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from pprint import pprint

for i in range(3):
    print()

print("C H A T B O T")
print("S T A R T E D")
print()


def to_splits(loader):
    docs = loader.load()
    for doc in docs:
        print(f"Loaded Document: {doc.metadata.get('source')}")
        print(f"Length: {len(doc.page_content)} ")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    all_splits = text_splitter.split_documents(docs)
    print(f"Chunks: {len(all_splits)}")
    print()
    return all_splits

WEBSITE = "https://www.futurebrains.io/team/nicolas-a-durr"
loader = WebBaseLoader(WEBSITE)
all_splits = to_splits(loader)



# Initialize an empty vector store
vectorstore = Chroma(collection_name="robins_collection", embedding_function=OpenAIEmbeddings())
vectorstore.add_documents(documents=all_splits)

loader = TextLoader("nick.txt")
all_splits = to_splits(loader)
vectorstore.add_documents(documents=all_splits)

loader = TextLoader("robin.txt")
all_splits = to_splits(loader)
vectorstore.add_documents(documents=all_splits)

print("Stored Chunks in Vector Store\n")

print("Chatbot: Hi! I'm your AI chatbot. Type 'bye' to exit.")

while True:
    user_input = input("You: ")

    if user_input.lower() == "bye":
        print("Chatbot: Goodbye!")
        break

    print()
    print(f"Lookup in Vectorstore, compare with Query and find 6 Nearest Neighbors\n")
    # Retrieve embeddings for a document based on its content or metadata
    query_result = vectorstore.similarity_search("Some query text", k=6)

    for i, res in enumerate(query_result):
        embeddings = vectorstore._embedding_function.embed_documents([res.page_content])
        print(f"{i+1} Embedding - Length: {len(embeddings[0])}")
        head_embedding = embeddings[0][:5]
        print(f"{i+1} Embedding - Vector (first 5 dimensions): {head_embedding}")
        print()

    retriever = vectorstore.as_retriever(search_kwargs={"k": 6})

    m = "gpt-4o-mini"
    print(f"Use Model: {m}\n")
    llm = ChatOpenAI(model=m)

    print("Use the following PROMPT Template\n")

    sys_prompt = """You are an assistant for question-answering tasks.
        Use the following pieces of retrieved context to answer
        the question. If you don't know the answer, say that you
        don't know. Use three sentences maximum and keep the
        answer concise.
        answer in german.
        {context}"""

    pprint(sys_prompt)
    print()

    system_prompt = (
        sys_prompt

    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )

    store = {}
    def get_session_history(session_id: str):
        if session_id not in store:
            store[session_id] = ChatMessageHistory()  # or InMemoryChatMessageHistory()
        return store[session_id]



    # specifies how retrieved context is fed into a promp
    # stuff means no further processing like summarization
    question_answer_chain = create_stuff_documents_chain(llm, prompt)

    # adds the retrieval step and propagates the retrieved context through the chain
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)

    conversational_chain = RunnableWithMessageHistory(
        rag_chain,  # Replace with your actual chain
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    response = conversational_chain.invoke(
        {"input": user_input},
        config={"configurable": {"session_id": "your_session_id"}}
    )

    # response["answer"]: Plain answer
    print(f"Answer: {response['answer']}")

    print("\nSources of the Answer:")
    # response["context"]: List of Source documents
    for i, document in enumerate(response["context"]):
        print(f"{i+1}. Source: {document}")
        print()

