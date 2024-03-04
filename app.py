from dotenv import load_dotenv
import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI

def main():
     load_dotenv()
     st.set_page_config(page_title="Chat with your PDF")
     st.header("Chat with your PDF 🐱‍🏍")
     #Uploading the file
     pdf = st.file_uploader("Upload your pdf here!")
    # Extracting the text from file
     if pdf is not None:
          pdf_reader = PdfReader(pdf)
          text=""
          for page in pdf_reader.pages:
               text+=page.extract_text()

          #Split into chunks
          text_splitter = CharacterTextSplitter(
               separator="\n",
               chunk_size=1000,
               chunk_overlap=200,
               length_function=len
          )

          chunks=text_splitter.split_text(text)          
          #Create embeddings
          embeddings = OpenAIEmbeddings()
          knowladge_base = FAISS.from_texts(chunks,embeddings)

          user_question = st.text_input("Ask your question related to your PDF")
          if user_question:
               docs = knowladge_base.similarity_search(user_question)
               llm=OpenAI()
               chain = load_qa_chain(llm,chain_type="stuff")
               response=chain.run(input_documents=docs,question=user_question)

               st.write(response)

                

if __name__ == "__main__":
     main()