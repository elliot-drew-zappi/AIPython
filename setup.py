from setuptools import setup, find_packages

setup(
    name="aipython",
    version="0.1.0",
    url="https://github.com/elliot-drew-zappi/AIPython",
    author="Elliot Drew",
    author_email="elliot.drew@zappistore.com",
    description="ChatGPT in the python console/notebooks with LangChain and some rich pretty formatting.",
    long_description=open('README.md').read(),
    packages=find_packages(),
    install_requires=[
        "openai",
        "langchain",
        "chromadb",
        "wikipedia",
        "unstructured",
        "rich",
        "tenacity",
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

