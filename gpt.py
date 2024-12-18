#!/usr/bin/env python3
import os
from openai import OpenAI
client = OpenAI(
   api_key=os.environ.get("OPENAI_API_KEY")
)

model = "gpt-4o"
prompt = (model.split("-")[0] + "-" + model.split("-")[1]).upper() + ":"

messages = []
interactive = False
temperature = 0.5
max_tokens = 2048

# Read arguments
import sys
input_message = ""

while len(sys.argv) > 1:
    # Parse -i and --interactive, no argument
    if sys.argv[1] in ["-i", "--interactive"]:
        interactive = True
        del sys.argv[1]
    # Parse -s and --system, string
    elif sys.argv[1] in ["-s", "--system"]:
        messages.append({"role": "system", "content": sys.argv[2]})
        del sys.argv[1:3]
    # Parse -t and --temperature, float between 0 and 1
    elif sys.argv[1] in ["-t", "--temperature"]:
        temperature = float(sys.argv[2])
        assert 0 <= temperature <= 2
        del sys.argv[1:3]
    # Parse -M and --max_tokens, integer
    elif sys.argv[1] in ["-M", "--max_tokens"]:
        max_tokens = int(sys.argv[2])
        del sys.argv[1:3]
    # Parse -p and --prompt, string
    elif sys.argv[1] in ["-p", "--prompt"]:
        prompt = sys.argv[2]
        del sys.argv[1:3]
    else:
        input_message += sys.argv[1]
        del sys.argv[1]

if input_message:
    messages.append({"role": "user", "content": input_message})

def gpt(messages):
    response = ""
    stream = client.chat.completions.create(
        model = model,
        messages = messages,
        temperature = temperature,
        max_tokens = max_tokens,
        stream=True
    )
    for chunk in stream:
        chunk = chunk.choices[0].delta.content or ""
        if chunk:
            response += chunk
            print(chunk, end="")

    return response

if interactive:
    while True:
        message = input("You: ")
        if message == "quit":
            break
        messages.append({"role": "user", "content": message})
        print()
        response = gpt(messages)
        print()
        messages.append(response)
else:
    if len(messages) == 1:
        print("Please provide a query or start an interactive session with -i.")
        sys.exit(1)

    response = gpt(messages)
    print()  # newline at the end
    # messages.append(response)
