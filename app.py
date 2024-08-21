import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings  # we will be embedding it using OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import OpenAI


def main():
    load_dotenv()
    # print(os.getenv("OPENAI_API_KEY"))  # to check if it's correctly accepting the OPEN_API_KEY
    st.set_page_config(
        page_title="Ask your PDF",
        page_icon=":page_with_curl:",
        layout="wide",
    )
    st.header("Ask your PDF 💭")
    pdf = st.file_uploader("Upload Your PDF", type=['pdf'])

    # extract the text

    if pdf is not None:  # checks if pdf exists or not
        pdf_reader = PdfReader(pdf)
        text = ""  # since PdfReader only reads one page at a time, we will loop through the pages
        for page in pdf_reader.pages:
            text += page.extract_text()  # we will extract text from each page and concatenate them
        # st.write(text)

        # separate it into chunks

        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,  # every chunk is of 1000 size
            chunk_overlap=200,  # it overlaps by 200 size of string everytime it starts at a new chunk
            length_function=len  # tells the way it checks for size, here we use len(in python)
        )
        chunks = text_splitter.split_text(text)  # Split incoming text and return chunks.
        # st.write(chunks)

        # embedding chunks and creating knowledge_base

        embeddings = OpenAIEmbeddings()  # using embeddings from OpenAI
        knowledge_base = FAISS.from_texts(chunks, embeddings)  # using FAISS along with embedding
        # to enable semantic search on the knowledge base

        # accept input

        user_question = st.text_input("Ask a question about your PDF: ")
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            # st.write(docs)

            llm = OpenAI()
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)
            st.write(response)

if __name__ == '__main__':
    main()