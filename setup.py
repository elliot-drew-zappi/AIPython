from setuptools import setup, find_packages

setup(
    name="aipython",
    version="0.1.1",
    url="https://github.com/elliot-drew-zappi/AIPython",
    author="Elliot Drew",
    author_email="elliot.drew@zappistore.com",
    description="ChatGPT in the python console/notebooks with LangChain and some rich pretty formatting.",
    long_description=open('README.md').read(),
    packages=find_packages(),
    install_requires=[
        "openai==0.27.2",
        "langchain==0.0.115",
        "chromadb==0.3.11",
        "wikipedia==1.4.0",
        "unstructured==0.5.4",
        "rich==13.0.1",
        "tenacity==8.2.2",
    ],
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
)

