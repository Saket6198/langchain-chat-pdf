# LangChain Ask Your PDF 

![image](https://github.com/user-attachments/assets/a8ec5ec7-f3ff-4424-930e-0d6d9c3801c9)


This is a Python application that allows you to upload a PDF and ask questions about it using Natural Language. This application uses a LLM to generate a response about your PDF. The LLM will not answer questions unrelated to the document.

## How it works

![Screenshot 2024-08-19 004903](https://github.com/user-attachments/assets/7b35ea75-2ddb-4843-a176-635ba015e95f)
The applications reads the PDF and divides the text into smaller readable chunks of fixed size using the langchain splitter which is then fed into the LLM. It uses OpenAI embedding to create a vector representation of the chunks. The application then finds the chunks that are semantically similiar to the question that the user asked and feeds those chunks to the LLM to generate a response.


Using the FAISS (Facebook AI Similarity Search), a powerful library designed for efficient similarity search and clustering of dense vectors thus creating a centralized knowledge base from which all the query's are processed.

## Installation

To install the repository, please clone this repository and install the requirements:

```
pip install -r requirements.txt
```

You will also need to add your OpenAI API key to the `.env` file.

## Usage

To use the application, run the `main.py` file with the streamlit CLI (after having installed streamlit): 

```
streamlit run app.py
```
