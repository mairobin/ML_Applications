from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain_core.messages import AIMessage

from key_reader import langchain_key, tavily_key, openai_key


"""
Problem: The model on its own does not have any concept of state. 

A HumanMessage returns an AIMessage, but a Follow Up Question is failing.

Human: I am Bob
AI: Hi Bob.
Human:  Whats my Name? 
AI: Sorry, I dont know


Solution: Pass the entire Conversation as a List of Messages to the Model when calling invoke()
"""

model = model = ChatOpenAI()

# Human Messages -> Question
# AI Message -> Answer

model.invoke(
    [
        HumanMessage(content="Hi! I'm Bob"),
        AIMessage(content="Hello Bob! How can I assist you today?"),
        HumanMessage(content="What's my name?"),
    ]
)












