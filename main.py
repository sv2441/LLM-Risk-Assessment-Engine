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


def generate_use_case_summary(df):
    qa_list = df.set_index('Questions').to_dict()['Answer']
    title_template = """
    Create an LLM downstream use case context with all of the following information.
         "{qa_list}"
                """
    prompt = ChatPromptTemplate.from_template(template=title_template)
    messages = prompt.format_messages(qa_list=qa_list)
    response = chat_llm(messages)
    use_case_summary = response.content
    
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

def main():
        
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
        st.write(f"""Provide a brief about the {use_case_title}.""")
        questions = [
        'What is the industry of the use case?',
        'What is the nature of the use case?',
        'What type of data is used?',
        'What Language Model (LLM) is used?',
        'What embedding approach is used?',
        'What embedding model is used?',
        'Who is the user group?',
        'Is fine-tuning applied?',
        'What is the domain context?',
        'What vector store is used?',
        'Is prompt engineering applied?',
        'How is the model deployed?',
        'How is the model monitored?',
        'What logging and feedback mechanisms are used?',
            'What guardrails are applied?'
        ]
        df = pd.DataFrame(questions, columns=['Questions'])
        df['Answer'] = ''

        config = {
            'Answer' : st.column_config.TextColumn('Answer (required)', width='large', required=True),
        }

        result = st.data_editor(df, column_config = config, num_rows='dynamic')
        result.to_csv('result_1.csv', index=False)
        
        if st.button('Get results'):
            st.write(result)
            st.session_state['result'] = result  # Store result in session state

        if 'result' in st.session_state:
            if st.session_state['result']['Answer'].isnull().any():
                st.write("Please answer all questions before continuing.")
            elif st.button("Continue"):
                with st.spinner('Processing your use_cases...'):
                    data=pd.read_csv('result_1.csv')
                    use_case_summary = generate_use_case_summary(data)
                    st.session_state['use_case_summary'] = use_case_summary  # Store use_case_summary in session state
                    st.write(use_case_summary)

        if 'use_case_summary' in st.session_state:
            if st.button("Generate Risk Assessment Document"):
                with st.spinner('Generating Risks...'):
                    risk_information = generate_risks(st.session_state['use_case_summary'])  # Use use_case_summary from session state
                    st.session_state['risk_information'] = risk_information  # Store risk_information in session state
                    st.subheader("Risk with Examples")
                    st.write(risk_information)

                with st.spinner('Ranking risks...'):
                    risk_ranking = rank_risks(st.session_state['risk_information'], use_case_title)  # Use risk_information from session state
                    st.session_state['risk_ranking'] = risk_ranking  # Store risk_ranking in session state
                    st.subheader("Risk Ranking")
                    st.write(risk_ranking)

                with st.spinner('Generating actionables for risk mitigation...'):
                    actionables = mitigate_risks(st.session_state['risk_ranking'])  # Use risk_ranking from session state
                    st.session_state['actionables'] = actionables  # Store actionables in session state
                    st.write(actionables)

                with st.spinner('Compiling the final document...'):
                    final_document = generate_summary(ps2_doc)
                    ps2_doc.save("PS2 Doc.docx")
                    filename = f"{use_case_title} Risk Assessment.docx"
                    save_docx(final_document, filename)
                    download_link = create_download_link(filename)
                    st.session_state['download_link'] = download_link  # Store download_link in session state
                    st.markdown(st.session_state['download_link'], unsafe_allow_html=True)  # Use download_link from session state
                    



if __name__ == "__main__":
    main()


