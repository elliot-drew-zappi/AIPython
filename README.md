# AIPython

## Intro

I got sick of copy and pasting stuff into the ChatGPT website and back into my editor/terminal/whatever, so I have made a small python module that I can use in the places I'm working most often - a python console or a python notebook.

The package uses [LangChain](https://github.com/hwchase17/langchain) heavily. It also uses [rich](https://github.com/Textualize/rich) to make the messages sent back prettier.

It uses the newish `gpt-3.5-turbo` model that powers ChatGPT for all queries. I haven't added cost tracking, but this will be something I do in the near future.

It is a WIP that I will add to as I go. I have not written any tests, because this is meant to be fun 💅.

On that note - I give no guarantees any of this will work. Use it at your own peril.

## Installation

Easiest thing to do is to install the package using pip straight from git:

```bash
pip install git+https://github.com/elliot-drew-zappi/AIPython.git
```

I'd recommend doing it into a virtualenv.

You will need your OpenAI API key and [organisation ID](https://platform.openai.com/account/org-settings) to be set as environmental variables. The org ID is useful when you want to charge the company you work for, rather than yourself, if you are using this for work. 

The API key will need to be named `OPENAI_USER_KEY` and the org ID `OPENAI_ORG_KEY`.

Next, if you want to use the VectorStore/Chroma document search facility, you need to set a path for the database and documents to be stored in an environmental variable called `AIPYTHON_DATA`. This needs to be somewhere the package can read and write to. Easiest bet would be to create a directory somewhere in your home directory and use that. When you first import and create the AI, the database and folders for documents will be created.

Not setting `AIPYTHON_DATA` won't stop you from using the package, it will just stop you from using the vector DB search.

## Usage

### Basic usage

To create the AI agent:

```python
from aipython.aipython import createAI
```

Asking questions is done through the `ask` method. Wow!

```python
AI.ask("write a poem about dark souls.")
```

Which returned:
```
In a world of darkness and despair, Where demons roam and   
dragons glare, A hero rises, brave and bold, To face the    
challenges that unfold.                                     

With sword in hand and shield held high, He ventures forth, 
prepared to die, For death is but a minor cost, In the land 
of Lordran, forever lost.                                   

The undead curse, a heavy weight, Upon his soul, a constant 
fate, But with each death, he learns and grows, Stronger,   
wiser, against his foes.                                    

The bosses loom, with deadly might, But our hero stands,    
ready to fight, With patience, skill, and a bit of luck, He 
strikes them down, with a final buck.                       

And so he journeys, through the dark, With fire and magic, a
deadly spark, Until at last, he reaches the end, And claims 
his prize, a hero's blend.                                  

Dark Souls, a game of trials and pain, But also triumph,    
glory, and gain, For those who dare to face the test, And   
prove themselves, among the best. 
```
The agent has memory, so you can ask them to elaborate on something they said before just like on the website:

```python
AI.ask("what is a 'heros blend'?")
''' returned:
In the context of the poem, "hero's blend" is a metaphorical
phrase that refers to the combination of qualities that make
a hero. It could be a blend of courage, determination,      
skill, and other virtues that enable the hero to overcome   
the challenges and emerge victorious. The phrase is not     
meant to refer to any specific tangible object or substance.
'''
```

### Passing in functions

A lot of the time I'm using AI to help me fix my garbage code - so I want to be able to pass in a function without any fuss.

You can pass a function you have defined using the `func` parameter to `.ask`. This will create a prompt that looks like: 

````{question}\n ```\n{function code}\n``` ````

So you could ask the AI to modify the function, write a test for the function, improve/critique/anything you can think of.

```python
def add(a, b):
   s = a + b
   return s

AI.ask("rewrite this function to be as complicated as possible:", func=add)
```

resulted in this:

Here is a more complicated version of the `add` function
```python
import numpy as np                                       
                                                         
def add(a, b):                                           
    if not isinstance(a, (int, float, complex, np.ndarray)):                                            
        raise TypeError(f"unsupported operand type(s) for +: '{type(a).__name__}' and '{type(b).__name__'")   
    if not isinstance(b, (int, float, complex, np.ndarray)):                                            
        raise TypeError(f"unsupported operand type(s) for +: '{type(a).__name__}' and '{type(b).__name__'")   
    if isinstance(a, np.ndarray) and isinstance(b, np.ndarray:                                             
        if a.shape != b.shape:                           
            raise ValueError(f"operands could not be broadcast together with shapes {a.shape} and {b.shape}") 
        return np.add(a, b)                              
    else:                                                
        s = a + b                                        
        return s   
```
This version of the `add` function includes type checking and 
error handling for unsupported operand types and            
incompatible shapes for numpy arrays. It also uses numpy's  
add function for adding numpy arrays. 

### Wikipedia

You can ask the AI to use wikipedia to answer questions. Just prefix your input to `ask` with `wiki:` to trigger the tool use. This is currently a seperate agent but the interaction messages will be added to the main agents conversational memory - so you can ask follow up questions about the wikipedia-assisted answer without the `wiki:` prefix.

### VectorStore search

If you have set the `AIPYTHON_DATA` environmental variable as above, you can use the vector db search. Similarly to the wiki search, this is done by prepending your input to `ask` with `vecdb:`. 

Of course, you need documents to search. At the path defined as `AIPYTHON_DATA`, you can create a directory called `vector_store` or initialise the AI agent with `createAI` which will also trigger creation of the dir. I have not fully tested this, but you can add basically any text file to this directory and it will be indexed and added to a Chroma vector index the next time you start up the AI. I think you might be able to add other types of files and it might figure out how to read them - but I have not tested it.

If you want to add more to the index, just pop more files in - next time it starts up it should check which files have not been added to the index yet and only add them.

Be warned that this does use the `ada` embeddings model from OpenAI so it does cost money... however its quite cheap in the grand scheme of things and this whole AI thing costs money.

### The `ai_exception` decorator

This decorator can be used on functions for debugging purposes. If a function runs normally with no errors - great, you get the normal results returned. However, if there is an exception, the decorator will take the function code and the traceback for the error and construct a prompt for the AI instructing it to try and fix the problem and send you the correct code.

Here is it in use:

```python
from aipython.aipython import ai_exception
   
@ai_exception(AI)
def divide(a, b):
    s = a / b
    return s

divide(1,2) # returns 0.5

divide(1, 0) # AI kicks in:
```
The error message indicates that the divide function is trying to divide by zero, which is not allowed in Python. To fix this      
error, you need to make sure that the second argument b is not zero before performing the division.                                

One way to do this is to add a check for b == 0 and raise a custom exception if b is zero. Here's an example:                      

```python
@ai_exception(AI)
def divide(a, b):  
    if b == 0: 
        raise ValueError("division by zero")   
    s = a / b
    return(s)
```

In this version of the divide function, we check if b is zero and raise a ValueError with a custom error message if it is. This    
will prevent the function from trying to divide by zero and causing a ZeroDivisionError. 

wow

### Classifier

You can use the `classify` method to do simple text classification tasks easily. Just pass in the text to be classified, and the labels for classification as a list of strings. 

There is a third parameter called `multi` which is `True` by default - pass this as `False` if you want to only get a single label out, otherwise the default might return multiple labels.

```python
AI.classify(
    "I don't really care about the whole thing tbh",
    [
        "Happy",
        "Sad",
        "Angry",
        "Bored",
        "Laughing",
        "Suprised",
        "Confused",
        "Intrigued",
        "Indifferent",
        "No emotion",
    ],
)
Indifferent          
```

### Conversation history

The conversation history is stored as `AI.conversation` as a list of dicts formatted as `{'question':"", 'answer':""}`.

This is useful if you want to go back in time and grab an answer you forgot about. It's also useful when you just want to print things in non-pretty markdown formatting - when you want to just paste in code the AI has produced for example.

In that case, just `print` the relevant answer.

Or if you want to pretty print the markdown again, `from aipython.agents.chat_agent import rich_print_md` will get you the function I use to do the pretty printing. Just pass in the markdown.

I will probably add a flag to `ask` to use pretty formatting or not in future.

## LICENSE

MIT license see LICENSE.txt
