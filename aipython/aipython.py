from aipython.agents.chat_agent import AIpython
import traceback
import inspect

# little decorator for ai debugging of functions
def ai_exception(AI):
    def decorator(func):
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                tb = traceback.format_exc()
                function_code = inspect.getsource(func)
                formatted_string = f"Here is a function, and the error I get when i run it - How do I fix it?:\n{function_code}\n\nTrackback:\n\n{tb}"
                AI.ask(formatted_string)
        return wrapper
    return decorator

def createAI():
    try:
        AI = AIpython()
        return(AI)
    except Exception as e:
        traceback.print_exc()
    

def main():
    print('''Import createAI() in a python script, notebook, python console etc and use it to instantiate the AIPython object.''')

if __name__ == "__main__":
    main()