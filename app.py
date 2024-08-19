import streamlit as st
import os
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

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


if __name__ == '__main__':
    main()
