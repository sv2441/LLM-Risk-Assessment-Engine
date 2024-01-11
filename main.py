import streamlit as st
import docx
import openai
from datetime import datetime
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv
from langchain.callbacks import get_openai_callback
import datetime
from langchain.chains import RetrievalQA
import io
import base64
from docx import Document
import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import numpy as np

load_dotenv()


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

chat_llm = ChatOpenAI(temperature=0.0)

embedding = OpenAIEmbeddings()

ps1_doc = docx.Document()
ps2_doc=docx.Document()

def read_docx(file_path):
    """
    Read the content of a docx file and return it as a text string.
    """
    doc = docx.Document(file_path)
    full_text = []
    for para in doc.paragraphs:
        full_text.append(para.text)
    return '\n'.join(full_text)

def save_docx(content, filename):
    # Create a new Document object
    doc = Document()

    # Add content to the document
    for paragraph in content.split('\n'):
        doc.add_paragraph(paragraph)

    # Save the document
    doc.save(filename)

# Function to create a download link
def create_download_link(filename):
    with open(filename, "rb") as file:
        data = file.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/docx;base64,{b64}" download="{filename}">Download Risk Assessment Document</a>'
    return href


def generate_questions(use_case_title): 
    
    st.subheader("Read and Answer the Below Questions")
    
    file_path = r'Questions to ask summary.docx'
    questions_to_ask_summary = read_docx(file_path)
     
    title_template = """
            Ask about the following in questions form questions based on the "{usecase}" provided.
            Each Queston should have its in deatil description related to the "{usecase}"Given . 
            Use "{questions_to_ask_summary}" document as a knowledge base for Generation of Question and in detail description for  "{usecase}". 
            Step 1: Collecting Basic Information
             1) Nature of Use Case: 
             2) Number of User Interactions:
             3) Purpose of Use Case: 
             Step 2: Understanding User Group and Sensitivity
             4) Intended User Group:
             5) Sensitivity of Use Case and Data:
             Step 3: Technical Details of LLM Implementation
             6) Nature of LLMs Used:
             7) Embedding Approach: 
             8) Vector Stores:
             9) Prompting Approach:
             10) Fine-Tuning Approach:
             11) Type of Evaluations: 
             12) Guardrails:
             13) Monitoring Approach:
             14) Deployment Model:
             15) Humans in the Loop:
             16) Logging and Feedback Mechanism:
                """
    prompt = ChatPromptTemplate.from_template(template=title_template)
    messages = prompt.format_messages(usecase=use_case_title, questions_to_ask_summary=3)
    response = chat_llm(messages)
    # doc.add_paragraph("Questions:")
    # doc.add_paragraph(response.content)
    questions=response.content
    return response.content


def generate_use_case_summary():
    use_case_summary="""
    Use Case Summary:
    
    Function: The chatbot is designed for answering FAQs related to customer queries.
    User Interactions: It is expected to handle approximately 200 user interactions per month.
    User Group: The primary users are the general public.
    Data Sensitivity: The chatbot does not handle sensitive or Personally Identifiable Information (PII).
    Deployment: The chatbot is deployed on a website hosted on Azure.
    LLM Used: OpenAI's GPT model.
    Database: Pinecone database is used.
    Embedding Approach: OpenAI embeddings are utilized.
    Prompting Approach: A simple prompting method is employed.
    Fine-Tuning: The model has been fine-tuned on a specific dataset.
    Evaluation: No formal evaluation has been done post-deployment.
    Guardrails for Misuse: No specific measures mentioned for preventing misuse.
    Monitoring: The chatbot is monitored by humans.
    Deployment Model: Cloud-based deployment on Azure.
    Human-in-the-Loop: There is no human-in-the-loop system for real-time oversight or intervention.
    Logging and Feedback: No details provided on logging and feedback mechanisms.
    """
    ps2_doc.add_paragraph("Use Case Summary:")
    ps2_doc.add_paragraph(use_case_summary)
    return use_case_summary

def generate_risks(use_case_summary):
    
    file_path = r'Documented nature of risks.docx'
    documented_nature_of_risks = read_docx(file_path)
    
    title_template = """
        Identify the risks that apply to the use case. Use the information in Knowledge base to identify the applicable risks. 
        Provide atleast 1 or 2 examples of each risk using the use case brief and the user responses to the questions.
        Knowledge base: {context}
        use case: {question}
                """
    prompt = ChatPromptTemplate.from_template(template=title_template)
    messages = prompt.format_messages(question=use_case_summary, context=documented_nature_of_risks)
    response = chat_llm(messages)
    risk_information = response.content
    ps2_doc.add_paragraph("Key Risks:")
    ps2_doc.add_paragraph(risk_information)
    return risk_information

def rank_risks(risk_information,use_case_title):
    
    # questions_and_answers = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    
    title_template = """
                Rank the "{risk_information}" in terms of priority and provide a criticality score as high/ medium/ low given for "{use_case_title}".
                It should have Criticality Score and Reason for the above "{risk_information}".
                create Table containing key risks with their risk ranking along with the reasons for the risk ranking.
                """
    prompt = ChatPromptTemplate.from_template(template=title_template)
    messages = prompt.format_messages(risk_information=risk_information,use_case_title=use_case_title)
    response = chat_llm(messages)
    risk_ranking = response.content
    ps2_doc.add_paragraph("Risk Ranking:")
    ps2_doc.add_paragraph(risk_ranking)
    return risk_ranking

def mitigate_risks(risk_ranking):
    st.subheader("Actionables for Risk Mitigation")
    title_template = """Provide Actionable steps for governance to address each identified risk for "{risk_ranking}".
        For each risk compile a set of actionables to address the "{risk_ranking}". These actionables shall be governance actionables.
        
    """
    prompt = ChatPromptTemplate.from_template(template=title_template)
    messages = prompt.format_messages(risk_ranking=risk_ranking)
    response = chat_llm(messages)
    actionables = response.content
    ps2_doc.add_paragraph("Actionables:")
    ps2_doc.add_paragraph(actionables)
    return actionables

def generate_summary(doc):
    st.subheader("Final Summary")
    summary = "\n".join(paragraph.text for paragraph in doc.paragraphs)
    
    title_template = """Compile All information in "{summary}". Structure in the following format:
        The document shall contain the following information: 
        Section A: Brief about the use case. 
        Section B: List of high-level risks associated with the use case.
        Section C: Table containing key risks with their risk ranking along with the reasons for the risk ranking.
        Section D: List of actionables for each risk listed in Section C.
    """
    prompt = ChatPromptTemplate.from_template(template=title_template)
    messages = prompt.format_messages(summary=summary)
    response = chat_llm(messages)
    final_document = response.content
    return final_document
   
#UI

st.title("LLM Risk Assessment Engine")

# Print the provided text
st.write("""
The LLM Risk Assessment Engine is designed to specialize in risk evaluation, providing insights and assessments for a range of risks including content risks, context risks, trust risks, and societal and sustainability risks.
""")
# Step 1: Get the use case title from the user
use_case_title = st.text_input("Write Your Use Case")

# Step 2: Ask Questions
# Step 3 : Get Answer
if use_case_title:
    st.write(f"""Provide a brief about the {use_case_title}. 
             The brief shall contain nature of use case, Nature of data, User group, Purpose of use case, Approach towards Embedding/ Prompt engineering/ Fine tuning / Evaluation, Guardrails applied, Role of Human in the loop, logging and feedback mechanism, sensitivity of use case and deployment model """)
    user_answer_1 = st.text_area('Write your use case in brief', height=150)

    # Step 4: Ask Question again which is not answered
    # step 5 : Answer Remainng Question
    if user_answer_1:
        with st.spinner('Generating questions...'):
            # doc.add_paragraph("Title:")
            # doc.add_paragraph(use_case_title)
            
            question = generate_questions(use_case_title)
        st.write(question)
        user_answer_2 = st.text_area('Write your use case based on question asked', height=200)
        # step 6: Show Use Case Summary 
        if user_answer_2:
            with st.spinner('Processing your use cases...'):
                use_case_summary=generate_use_case_summary()
                st.write(use_case_summary)
                
                
####################################################################################################################

                
                if st.button("Generate Risk Assessment Document"):
                # step 7 : button for generation risk , rank , actionables and final document
                    with st.spinner('Generating Risks...'):
                        # doc.add_paragraph("Answers:")
                        # doc.add_paragraph(user_answer)
                        risk_information = generate_risks(use_case_summary)
                        st.subheader("Risk with Examples")
                    st.write(risk_information)

                    with st.spinner('Ranking risks...'):
                        risk_ranking = rank_risks(risk_information,use_case_title)
                        st.subheader("Risk Ranking")
                    st.write(risk_ranking)

                    with st.spinner('Generating actionables for risk mitigation...'):
                        actionables = mitigate_risks(risk_ranking)
                    st.write(actionables)

                    with st.spinner('Compiling the final document...'):
                        final_document = generate_summary(ps2_doc)
                        ps2_doc.save("PS2 Doc.docx")
                        filename = f"{use_case_title} Risk Assessment.docx"
                        save_docx(final_document, filename)
                        download_link = create_download_link(filename)
                    st.markdown(download_link, unsafe_allow_html=True)

