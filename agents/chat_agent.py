import os
import json
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
import re

import openai
openai.api_key = os.getenv("OPENAI_USER_KEY")
openai.org = os.getenv("OPENAI_ORG_KEY")

from langchain.agents import Tool
from langchain.memory import ConversationBufferWindowMemory
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, ConversationChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder
)
from langchain.utilities import WikipediaAPIWrapper
from langchain.agents import initialize_agent

c = Console()

def create_wiki_agent():
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=3)
    llm=ChatOpenAI(model_name='gpt-3.5-turbo',temperature=0.)
    wiki = WikipediaAPIWrapper()
    tools = [
        Tool(
            name = "Wikipedia Search",
            func=wiki.run,
            description="useful for when you need to answer questions about general knowledge (history, science, maths, geography, culture, technology etc). the input to this should be a single search term. Your answers should be in Markdown."
        ),
    ]
    agent_chain = initialize_agent(
        tools, 
        llm, 
        agent="chat-conversational-react-description", 
        verbose=False,
        memory=memory,
        )

    return agent_chain

def create_chat():
    template="You are a helpful assistant that is a genius at programming. You respond with markdown formatted text. If you create a markdown fenced code block for a specific programming language, you must indicate what syntax highlighting to use e.g. ```python" 
    human_template="{input}"
    chat_prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(template),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template(human_template)
        ])

    chat = ChatOpenAI(
        model_name='gpt-3.5-turbo',
        temperature=0,
        max_tokens= 1000
        )
    chat_memory = ConversationBufferWindowMemory(return_messages=True, k=5)
    chain = ConversationChain(llm=chat, prompt=chat_prompt, memory=chat_memory)
    return(chain)

def parse_markdown_chunks(markdown_text):
    chunks = []
    lines = markdown_text.split("\n")
    in_table = False
    current_chunk = []

    for line in lines:
        if line.startswith("|") and not in_table:
            if current_chunk:
                chunks.append("\n".join(current_chunk))
                current_chunk = []
            in_table = True
        elif not line.startswith("|") and in_table:
            chunks.append("\n".join(current_chunk))
            current_chunk = []
            in_table = False

        current_chunk.append(line)

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

def markdown_table_to_rich_table(table_chunk):
    lines = table_chunk.split("\n")
    header_line = lines[0]
    delimiter_line = lines[1]

    headers = [header.strip() for header in header_line.split("|") if header.strip()]
    columns = len(headers)

    table = Table()
    for header in headers:
        table.add_column(header)

    for line in lines[2:]:
        row = [cell.strip() for cell in line.split("|") if cell.strip()]
        if len(row) == columns:
            table.add_row(*row)

    return table

def rich_print_md(text):
    chunks = parse_markdown_chunks(text)
    for chunk in chunks:
        if chunk.startswith("|"):
            rich_table = markdown_table_to_rich_table(chunk)
            c.print(rich_table)
        else:
            c.print(Markdown(chunk))


class AIpython:
    def __init__(self):
        self.conversation = []
        self.wiki_agent = create_wiki_agent()
        self.code_chat = create_chat()

    def ask_wiki(self, question):
        r = self.wiki_agent.run(input = question)
        #json.dumps(r, indent=2)
        c.print(Markdown(r))

    def ask(self, question):
        with c.status("[bold green]Answering Question...", spinner='aesthetic', speed=0.8) as status:
            m = self.code_chat.predict(input = question)
            self.conversation.append({'question':question, 'answer':m})
            rich_print_md(m)
        return m

    