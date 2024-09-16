# Read the contents of the text file
with open('langchain_keys.txt', 'r') as file:
    for line in file:
        # Execute the line as a Python assignment
        exec(line.strip())

print()