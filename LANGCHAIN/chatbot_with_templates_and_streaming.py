from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage


from langchain_core.chat_history import (
    BaseChatMessageHistory,
    InMemoryChatMessageHistory,
)
from langchain_core.runnables.history import RunnableWithMessageHistory

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.messages import SystemMessage, trim_messages

from key_reader import langchain_key, tavily_key, openai_key


"""
This Script builds a Chatbot
    - considers chat history
    - using different Templates


Problem: The model on its own does not have any concept of state. 

A HumanMessage returns an AIMessage, but a Follow Up Question is failing.

Human: I am Bob
AI: Hi Bob.
Human:  Whats my Name? 
AI: Sorry, I dont know


Solution: Pass the entire Conversation as a List of Messages to the Model when calling invoke()

But Attention: 
    - List of Messages can overflow context window
    - Act before prompt template but after loading previous messages
    - build in helpers. e.g. trim_messages specifies how many tokens to keep


"""

model = model = ChatOpenAI()

# Human Messages -> Question
# AI Message -> Answer


# The response is an AI message and returns Bob
response = model.invoke(
    [
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content="Hello Bob! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
)


store = {}

# session id - to distinguish between different sessions
# different session leads to different histories
# returns history in an in memory object
#
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = InMemoryChatMessageHistory()
    return store[session_id]

# A runable is a unit of work that can be invoked, batched, streamed, transformed and composed.
# runables are the things than can be chained into sequences
with_message_history = RunnableWithMessageHistory(model, get_session_history)

# config contains session as info for the runnable
# is not part of the input

config = {"configurable": {"session_id": "abc2"}}

response = with_message_history.invoke(
    [HumanMessage(content="Hi! I'm Bob")],
    config=config,
)

response.content

# Follow up Questions follow the same approach
# Different sessions lead to different histories
# We can always go back to the original conversation (since we are persisting it in a database)
response = with_message_history.invoke(
    [HumanMessage(content="What's my name?")],
    config=config,
)

response.content


"""
# Prompt Templates
# Purpose:  Extend the raw user information to improve quality of Replies

Example: System Messages
    - are send to LLM to perform task better, but are not part of user input
    - e.g. "You are a helpful assistant. Answer all questions to the best of your ability."

MessagesPlaceholder
    - what messages to be rendered during formatting. 
    - useful when 
        - you are uncertain of what role you should be using for your message prompt templates
        - when you wish to insert a list of messages during formatting
"""



prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer all questions to the best of your ability.",
        ),
        MessagesPlaceholder(variable_name="messages"), # used for Message History
    ]
)

chain = prompt | model

response = chain.invoke({"messages": [HumanMessage(content="hi! I'm bob")]})

response.content


trimmer = trim_messages(
    max_tokens=65,
    strategy="last",
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

messages = [
    SystemMessage(content="you're a good assistant"),
    HumanMessage(content="hi! I'm bob"),
    AIMessage(content="hi!"),
    HumanMessage(content="I like vanilla ice cream"),
    AIMessage(content="nice"),
    HumanMessage(content="whats 2 + 2"),
    AIMessage(content="4"),
    HumanMessage(content="thanks"),
    AIMessage(content="no problem!"),
    HumanMessage(content="having fun?"),
    AIMessage(content="yes!"),
]

# Wont remember the Name as it is trimmed
trimmer.invoke(messages)



"""
Streaming for USER EXPERIENCE

Easy to implemenet

"""
config = {"configurable": {"session_id": "abc15"}}
for r in with_message_history.stream(
    {
        "messages": [HumanMessage(content="hi! I'm todd. tell me a joke")],
        "language": "English",
    },
    config=config,
):
    print(r.content, end="|")









