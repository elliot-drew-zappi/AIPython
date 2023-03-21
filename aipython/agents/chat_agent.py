import os
import json
from rich.console import Console
from rich.markdown import Markdown
from rich.table import Table
import re
import inspect

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

from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.document_loaders import DirectoryLoader

import traceback

# chroma vectorstore
def setup_index():
    if os.getenv("AIPYTHON_DATA"):
        index_path=os.path.join(os.getenv("AIPYTHON_DATA"), "chromadb")
        doc_path=os.path.join(os.getenv("AIPYTHON_DATA"), "vector_store/")
        embedding = OpenAIEmbeddings()
        if os.path.exists(index_path):
            vectordb = Chroma(persist_directory=index_path, embedding_function=embedding)
            with open(os.path.join(os.getenv("AIPYTHON_DATA"), "chroma_seen.json"), "r") as fin:
                chromaseen = json.loads(fin.read())
            loader = DirectoryLoader(doc_path)
            documents = loader.load()
            new_documents = []
            for doc in documents:
                doc_name = documents[0].metadata['source']
                if doc_name not in chromaseen:
                    new_documents.append(doc)
                    chromaseen.append(doc_name)
            if len(new_documents) > 0:
                text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=25)
                docs = text_splitter.split_documents(new_documents)
                vectordb.add_documents(docs)
                with open(os.path.join(os.getenv("AIPYTHON_DATA"), "chroma_seen.json"), "w") as fout:
                    fout.write(json.dumps(chromaseen))
        else:
            if not os.path.exists(doc_path):
                os.makedirs(doc_path)
            loader = DirectoryLoader(doc_path)
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=25)
            docs = text_splitter.split_documents(documents)
            vectordb = Chroma.from_documents(docs, embedding, persist_directory=index_path)
            vectordb.persist()
            with open(os.path.join(os.getenv("AIPYTHON_DATA"), "chroma_seen.json"), "w") as fout:
                fout.write(json.dumps(os.listdir(doc_path)))
        return(vectordb)
    else:
        rich_print_md("**Not using VectorDB** - set AIPYTHON_DATA environmental variable to the path you want Chroma DB and vector files to be stored.")
        return(None)

def create_wiki_agent():
    memory = ConversationBufferWindowMemory(memory_key="chat_history", return_messages=True, k=5)
    llm = ChatOpenAI(
        model_name='gpt-3.5-turbo',
        temperature=0,
        max_tokens= 500
        )
    wiki = WikipediaAPIWrapper()
    tools = [
        Tool(
            name = "Wikipedia Search",
            func=wiki.run,
            description="you should use this tool only when the user starts their message with 'wiki:'. the input to this should be a single search term. You should start your answer with the phrase, 'Using wikipedia search: '"
        ),
    ]
    agent_wiki = initialize_agent(
        tools, 
        llm, 
        agent="chat-conversational-react-description", 
        verbose=True,
        memory=memory,
        )

    return agent_wiki

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
        max_tokens= 500
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
    c = Console()
    chunks = parse_markdown_chunks(text)
    for chunk in chunks:
        if chunk.startswith("|"):
            rich_table = markdown_table_to_rich_table(chunk)
            c.print(rich_table)
        else:
            c.print(Markdown(chunk))

def chunk_strings(s_list, chunksize = 3000):
    chunks = []
    for s in s_list:
        for i in range(0, len(s), chunksize):
            chunks.append(s[i:i+chunksize])
    return chunks

class AIpython:
    def __init__(self):
        self.conversation = []
        self.wiki_agent = create_wiki_agent()
        self.code_chat = create_chat()
        self.vector_db = setup_index()
        self.c = Console()
    
    def clear(self):
        self.wiki_agent.memory.clear()
        self.code_chat.memory.clear()

    def ask_wiki(self, question, plaintext):
        with self.c.status("[bold green]Answering Question...", spinner='aesthetic', speed=0.8) as status:
            m = self.wiki_agent.run(input = question)
            self.conversation.append({'question':question, 'answer':m})
            # we now add the memory to the code chat so it can use it as context.
            self.code_chat.memory.chat_memory.add_user_message(self.wiki_agent.memory.chat_memory.messages[0].content)
            self.code_chat.memory.chat_memory.add_ai_message(self.wiki_agent.memory.chat_memory.messages[1].content)
            # then clear the memory of the wiki agent
            self.wiki_agent.memory.clear()
        if plaintext:
            print(m)
        else:
            rich_print_md(m)

    def ask_normal(self, question, plaintext):
        with self.c.status("[bold green]Answering Question...", spinner='aesthetic', speed=0.8) as status:
            m = self.code_chat.predict(input = question)
            self.conversation.append({'question':question, 'answer':m})
        if plaintext:
            print(m)
        else:
            rich_print_md(m)
        
    def ask(self, question, func = None, plaintext = False):
        
        if func:
            try:
                func_source = inspect.getsource(func)
                question += f"\n```\n{func_source}\n```"
            except Exception as e:
                print(f"There was an error: {e}")
                return

        if question.startswith("wiki:"):
            try:
                self.ask_wiki(question, plaintext)
            except Exception as e:
                print(f"There was an error: {e}")
        elif question.startswith("vecdb:"):
            if self.vector_db:
                try:
                    docs = self.vector_db.similarity_search(question[5:])
                    # now feed these into normal as context.
                    new_input = f"Context:\n{docs[0].page_content}\n{docs[1].page_content}\n{docs[2].page_content}\n\nQuestion:{question[5:]}"
                    self.ask_normal(new_input, plaintext)
                except Exception as e:
                    print(f"There was an error: {e}")
            else:
                try:
                    rich_print_md("**Not using VectorDB** - set AIPYTHON_DATA environmental variable.")
                    self.ask_normal(question, plaintext)
                except Exception as e:
                    print(f"There was an error: {e}")
        else:
            try:
                self.ask_normal(question, plaintext)
            except Exception as e:
                print(f"There was an error: {e}")
    
    def classify(self, query, labels, multi = True):
        self.clear() # fresh, uncontaminated convo
        if multi:
            amount = "one or more"
        else:
            amount = "one"
        label_text = ", ".join(labels)
        prompt = f"Instruction: Classify the Query as {amount} of {label_text}.Only return the label, no other text.\nQuery: '{query}'"
        l = self.ask_normal(prompt, True)
        return l

    def summarize(self, docs, instruction = "Summarise the following, only answer with the summary:"):
        """
        Recursively sumarises any number of documents each of arbitrary length.
        docs is a list of strings
        instruction is a string
        returns a string summary
        """
        
        # first get chunks
        chunks = chunk_strings(docs)
        # now we need to start summarising and combining the chunks.
        while len(chunks) > 1:
            summaries = []
            for chunk in chunks:
                q = f"{instruction}\n{chunk}"
                self.clear()
                summary = self.ask(q)
                summaries.append(summary)
            # join up the summaries and pass in a list for chunking.
            chunks = chunk_strings(["\n".join(summaries)])
        q = f"{instruction}\n{chunks[0]}"
        self.clear()
        summary = self.ask(q)
        return(summary)
    