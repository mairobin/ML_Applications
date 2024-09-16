import os

# Read the contents of the text file
with open('langchain_keys.txt', 'r') as file:
    for line in file:
        # Execute the line as a Python assignment
        exec(line.strip())


# ENVIRONMENT VARIABLES
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'
os.environ['LANGCHAIN_API_KEY'] = langchain_key

os.environ['OPENAI_API_KEY'] = openai_key

os.environ['TAVILY_API_KEY'] = tavily_key

print("Environment Variables set successfully")

