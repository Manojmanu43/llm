import streamlit as st
import pickle
from PyPDF2 import PdfReader
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.callbacks import get_openai_callback
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
os.environ['OPENAI_API_KEY'] = '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'

def main():
    st.header('Chat with PDF document')
    st.sidebar.title('LLM ChatApp using LangChain')
    st.sidebar.markdown('''
    This is an LLM powered chatbot built using:
    - [Streamlit](https://streamlit.io/)
    - [LangChain](https://python.langchain.com/)
    - [OpenAI](https://openai.com/docs/models) LLM Model
    ''')

    #Upload a PDF File

    pdf = st.file_uploader("upload your PDF File",type='pdf')

    if pdf is not None:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text+=page.extract_text()
        #st.write(text)
        text_splitter=RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        chunks = text_splitter.split_text(text=text)
        #st.write(chunks[0])

        download_name = pdf.name[:-4]

        faiss_index_path = f"{download_name}.index"


        if os.path.exists(faiss_index_path):
            embeddings= OpenAIEmbeddings()
            VectorStore= FAISS.load_local(faiss_index_path,embeddings)
            st.write('Embeddings Loaded from the Disk')
        else:
            embeddings = OpenAIEmbeddings()
            VectorStore = FAISS.from_texts(chunks, embeddings)
            VectorStore.save_local(faiss_index_path)
            st.write('Embeddings Created')

        query = st.text_input("Enter your question from your PDF File")

        if query:
            docs = VectorStore.similarity_search(query=query,k=3)
            llm = OpenAI()
            chain = load_qa_chain(llm = llm, chain_type='stuff')
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs,question=query)
                print(cb)
            st.write(response)

if __name__ == '__main__':
    main()