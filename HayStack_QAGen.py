import streamlit as st
import os
import docx
import haystack
from haystack.document_stores import InMemoryDocumentStore
from haystack.nodes import BM25Retriever,TfidfRetriever,PromptNode, PromptTemplate
from haystack.nodes import FARMReader,TransformersReader
from haystack.pipelines import ExtractiveQAPipeline,DocumentSearchPipeline,GenerativeQAPipeline,Pipeline
from haystack.utils import print_answers
from haystack.pipelines.standard_pipelines import TextIndexingPipeline
from haystack import Document
from haystack.utils import print_answers
import pdfplumber
from haystack.nodes import OpenAIAnswerGenerator
from haystack.nodes.prompt import PromptTemplate
import re

document_store = InMemoryDocumentStore(use_bm25=True)
documents=[]
def add_document(document_store, file):
    if file.type == 'text/plain':
        text = file.getvalue().decode("utf-8")
        # document_store.write_documents(dicts)
        # st.write(file.name)
        # st.write(text)
    elif file.type == 'application/pdf':
        with pdfplumber.open(file) as pdf:
            text = "\n\n".join([page.extract_text() for page in pdf.pages])
            # document_store.write_documents([{"text": text, "meta": {"name": file.name}}])
            # st.write(text)
    elif file.type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
        doc = docx.Document(file)
        text = "\n\n".join([paragraph.text for paragraph in doc.paragraphs])
        # document_store.write_documents([{"text": text, "meta": {"name": file.name}}])
        # st.write(text)
    else:
        st.warning(f"{file.name} is not a supported file format")
    dicts = {
            'content': text,
            'meta': {'name': file.name}
            }  
    documents.append(dicts) 

    
# create Streamlit app
st.title("Contextualized Search for Document Archive")
API_KEY = st.secrets['OPENAI_API_KEY']
# create file uploader
uploaded_files = st.file_uploader("Upload Files", accept_multiple_files=True)

# loop through uploaded files and add them to document store
if uploaded_files:
    for file in uploaded_files:
        add_document(document_store, file)
document_store.write_documents(documents)
# display number of documents in document store
st.write(f"Number of documents uploaded to document store: {document_store.get_document_count()}")

if (document_store.get_document_count()!=0):
    question = st.text_input('Ask a question')
    retriever = TfidfRetriever(document_store=document_store)
     # QA pipeline using prompt node 
    if question != '':  
        my_template =  PromptTemplate(name="question-answering",
                             prompt_text="""Given the context and the given question,provide a clear and concise response from the relevant information presented in the paragraphs.
                             If the question cannot be answered from the context, reply with 'No relevant information present in attached documents'.
                             \n===\nContext: {examples_context}\n===\n{examples}\n\n
                             ===\nContext: {context}\n===\n{query}"""
            )
        node = OpenAIAnswerGenerator(
                api_key=API_KEY,
                model="text-davinci-003",
                max_tokens=150,
                presence_penalty=0.1,
                frequency_penalty=0.1,
                top_k=1,
                temperature=0,
                prompt_template=my_template
            )   
        candidate_documents = retriever.retrieve(query=question,top_k=2)
        pipe = Pipeline()
        pipe.add_node(component=node, name="prompt_node", inputs=["Query"])
        response = pipe.run(query=question, documents=candidate_documents)
        output=response["answers"]
        # answer = output[0].answer
        if output:
            answer = output[0].answer
            st.write(answer)
        else:
            st.write('Please try with another question')
    






        