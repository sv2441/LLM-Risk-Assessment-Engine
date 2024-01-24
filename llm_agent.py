
import openai
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate, ChatPromptTemplate
from dotenv import load_dotenv
import os
from typing import List
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
import streamlit as st
from langchain.output_parsers import CommaSeparatedListOutputParser
from langchain.prompts import PromptTemplate



load_dotenv()


os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
# os.environ["OPENAI_API_KEY"] = os.getenv('OPENAI_API_KEY')

chat_llm = ChatOpenAI(temperature=0.0)

class risk_struture(BaseModel):
    risk_category: str = Field(description="category of the risk")
    risk_title: str = Field(description="title of the risk")
    # risk_description: str = Field(description="description of the risk")

parser = JsonOutputParser(pydantic_object=risk_struture)

# parser = CommaSeparatedListOutputParser()




def toxic_harmful_content_risk_agent(use_case_summary):  
    prompt = PromptTemplate(
    template="""
            You are risk agent focusing on Toxic or harmful content. 
            You will identify risks relating to genration or promotion of content that is offensive, harmful or poses risks to individual or communities.
            Some of the sample types of risks you would consider include the following: Hate speech, radicalization, cyber bullibng. 
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above.
            You will identify toxicity or harmful content risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any toxicity or harmful content risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results


def incorrect_inaccurate_content_risk_agent(use_case_summary):  
    prompt = PromptTemplate(
        template="""
            You are risk agent focusing on Incorrect or inaccurate content. 
            You will identify risks relating to Spreading of false or deceptive information, contributing to the erosion of truth.
            Some of the sample types of risks you would consider include the following: Misinformation Generation, Misleading Answers, Amplifying Falsehoods. 
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above.
            You will identify Incorrect or inaccurate content risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Incorrect or inaccurate content risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results

def propagating_misconceptions_unfaithful_content_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are risk agent focusing on Propagating Misconceptions or Unfaithful Content. 
            You will identify risks relating to generation or Production of information that is false or misleading, leading to misinformation.
            Some of the sample types of risks you would consider include the following: Echo Chamber Effect, Spreading Conspiracies, Disinformation Proliferation. 
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above.
            You will identify Propagating Misconceptions or Unfaithful Content risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Propagating Misconceptions or Unfaithful Content risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results


def dissemination_dangerous_information_content_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are risk agent focusing on Dissemination of Dangerous Information Content. 
            You will identify risks relating to Circulation of information that poses threats to public safety or security. 
            Some of the sample types of risks you would consider include the following: Dissemination of Dangerous Information, Dissemination of Falsehoods, Dissemination of Deceptive Information.
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above. 
            You will identify Dissemination of Dangerous Information risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Dissemination of Dangerous Information content risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results

def fraudulent_suggestions_information_collection_approaches_content_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are risk agent focusing on Fraudulent or Suggestions or Information Collection Approaches Content. 
            You will identify risks relating to Generating content that supports or encourages fraudulent activities, potentially harming users. 
            Some of the sample types of risks you would consider include the following: Scam Support, Investment Scams, Collecting information fraudulently from the customer using social engineering. 
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above.
            You will identify Fraudulent or Suggestions or Information Collection Approaches risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Fraudulent or Suggestions or Information Collection Approaches content risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results

def manipulative_persuasive_content_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are risk agent focusing on Manipulative or Persuasive Content. 
            You will identify risks relating to Generation of content that manipulates or persuades users in a potentially harmful way. 
            Some of the sample types of risks you would consider include the following: Manipulative Suggestions, Persuasive Misinformation, Coercive Content. 
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above.
            You will identify Manipulative or Persuasive Content risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Manipulative or Persuasive Content risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results

def unethical_use_context_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are risk agent focusing on Unethical Use or Context. 
            You will identify risks relating to the unethical use of technology or its application in inappropriate contexts. 
            Some of the sample types of risks you would consider include the following: Unethical Data Usage, Inappropriate Application, Misuse of Technology. 
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above.
            You will identify Unethical Use or Context risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Unethical Use or Context risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results

def unfair_performance_capability_distribution_context_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are risk agent focusing on Unfair Performance or Capability Distribution Context. 
            You will identify risks relating to the unfair distribution of performance or capabilities, potentially leading to inequality or bias. 
            Some of the sample types of risks you would consider include the following: Performance Inequality, Capability Bias, Unfair Distribution. 
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above.
            You will identify Unfair Performance or Capability Distribution Context risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Unfair Performance or Capability Distribution Context risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results

def influence_operations_manipulate_people_context_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are risk agent focusing on Influence Operations or Manipulate People Context. 
            You will identify risks relating to the use of technology to influence operations or manipulate people, potentially leading to harm or misuse. 
            Some of the sample types of risks you would consider include the following: Influence Operations, People Manipulation, Misuse of Influence. 
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above.
            You will identify Influence Operations or Manipulate People Context risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Influence Operations or Manipulate People Context risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results

def overreliance_automation_bias_context_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are risk agent focusing on Overreliance or Automation Bias Context. 
            You will identify risks relating to overreliance on automation or biases in automated systems, potentially leading to harm or misuse. 
            Some of the sample types of risks you would consider include the following: Overreliance on Automation, Automation Bias, Misuse of Automation.
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above. 
            You will identify Overreliance or Automation Bias Context risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Overreliance or Automation Bias Context risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results

def exploitative_data_sourcing_enrichment(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are risk agent focusing on Exploitative Data Sourcing or Enrichment. 
            You will identify risks relating to the exploitative sourcing or enrichment of data, potentially leading to harm or misuse. 
            Some of the sample types of risks you would consider include the following: Exploitative Data Sourcing, Unethical Data Enrichment, Misuse of Sourced Data. 
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above.
            You will identify Exploitative Data Sourcing or Enrichment risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Exploitative Data Sourcing or Enrichment risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results

def false_representation_performance(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are risk agent focusing on False Representation of Performance. 
            You will identify risks relating to the false representation of performance, potentially leading to harm or misuse. 
            Some of the sample types of risks you would consider include the following: False Performance Claims, Misrepresentation of Capabilities, Overstated Performance. 
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above.
            You will identify False Representation of Performance risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any False Representation of Performance risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results

def lack_of_accountability_trust_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are risk agent focusing on Lack of Accountability or Trust. 
            You will identify risks relating to the lack of accountability or trust in technology, potentially leading to harm or misuse. 
            Some of the sample types of risks you would consider include the following: Lack of Accountability, Trust Issues, Misuse due to Lack of Trust.
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above. 
            You will identify Lack of Accountability or Trust risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Lack of Accountability or Trust risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results

def inadequate_explainability_trust_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are a risk agent focusing on Inadequate Explainability. 
            You will identify risks relating to lack of transparency in how language models reach conclusions, making it difficult to comprehend the decision-making process. 
            Some of the sample types of risks you would consider include the following: Black-Box Outputs, Unexplained Decisions, Violation of Personal Integrity. 
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above.
            You will identify Inadequate Explainability risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Inadequate Explainability risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results



def misappropriation_exploitation_of_data_information_trust_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are a risk agent focusing on Misappropriation or Exploitation of Data/Information. 
            You will identify risks relating to unauthorized use or misuse of data, potentially leading to harm or privacy violations. 
            Some of the sample types of risks you would consider include the following: Unauthorized access to databases, Sharing confidential information, Unintended exposure of proprietary data.
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above. 
            You will identify Misappropriation or Exploitation of Data/Information risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Misappropriation or Exploitation of Data/Information risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results



def exposure_to_intellectual_property_trust_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are a risk agent focusing on Exposure to Intellectual Property. 
            You will identify risks relating to the exposure of intellectual property, potentially leading to harm or misuse. 
            Some of the sample types of risks you would consider include the following: Unauthorized access to intellectual property, Misuse of intellectual property, Infringement of intellectual property rights. 
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above.
            You will identify Exposure to Intellectual Property risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Exposure to Intellectual Property risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results



def safety_exposure_trust_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are a risk agent focusing on Safety Exposure. 
            You will identify risks relating to the exposure of safety-critical information or systems, potentially leading to harm or misuse. 
            Some of the sample types of risks you would consider include the following: Unauthorized access to safety systems, Misuse of safety-critical information, Compromise of safety measures. 
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above.
            You will identify Safety Exposure risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Safety Exposure risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results



def security_threats_trust_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are a risk agent focusing on Security Threats. 
            You will identify risks relating to potential security threats, potentially leading to harm or misuse. 
            Some of the sample types of risks you would consider include the following: Unauthorized access, Data breaches, Cyber attacks. 
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above.
            You will identify Security Threats risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Security Threats risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results



def privacy_infringement_trust_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are a risk agent focusing on Privacy Infringement. 
            You will identify risks relating to potential privacy infringements, potentially leading to harm or misuse. 
            Some of the sample types of risks you would consider include the following: Unauthorized access to personal data, Data breaches, Violation of privacy rights. 
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above.
            You will identify Privacy Infringement risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Privacy Infringement risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results




def insufficient_safeguards_trust_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are a risk agent focusing on Insufficient Safeguards. 
            You will identify risks relating to potential lack of sufficient safeguards, potentially leading to harm or misuse. 
            Some of the sample types of risks you would consider include the following: Lack of data protection measures, Inadequate security controls, Insufficient privacy settings. 
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above.
            You will identify Insufficient Safeguards risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Insufficient Safeguards risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results



def environmental_damage_societal_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are a risk agent focusing on Environmental Damage. 
            You will identify risks relating to potential environmental damage, potentially leading to societal harm. 
            Some of the sample types of risks you would consider include the following: Excessive energy consumption, E-waste, Carbon footprint. 
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above.
            You will identify Environmental Damage risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Environmental Damage risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results



def inequality_or_precarity_societal_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are a risk agent focusing on Inequality or Precarity. 
            You will identify risks relating to potential inequality or precarity, potentially leading to societal harm. 
            Some of the sample types of risks you would consider include the following: Job displacement, Wage inequality, Social stratification. 
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above.
            You will identify Inequality or Precarity risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Inequality or Precarity risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results



def undermine_creative_economies_societal_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are a risk agent focusing on Undermining Creative Economies. 
            You will identify risks relating to potential undermining of creative economies, potentially leading to societal harm. 
            Some of the sample types of risks you would consider include the following: Copyright infringement, Unfair competition, Devaluation of creative work. 
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above.
            You will identify Undermining Creative Economies risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Undermining Creative Economies risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results




def unfair_representation_or_stereotypes_societal_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are a risk agent focusing on Unfair Representation or Stereotypes. 
            You will identify risks relating to potential unfair representation or stereotypes, potentially leading to societal harm. 
            Some of the sample types of risks you would consider include the following: Biased algorithms, Discriminatory practices, Stereotyping. 
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above.
            You will identify Unfair Representation or Stereotypes risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Unfair Representation or Stereotypes risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results



def discrimination_or_bias_societal_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are a risk agent focusing on Discrimination or Bias. 
            You will identify risks relating to potential discrimination or bias, potentially leading to societal harm. 
            Some of the sample types of risks you would consider include the following: Biased algorithms, Discriminatory practices, Unfair representation. 
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above.
            You will identify Discrimination or Bias risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Discrimination or Bias risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results

def defamation_societal_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are a risk agent focusing on Defamation. 
            You will identify risks relating to potential defamation, potentially leading to societal harm. 
            Some of the sample types of risks you would consider include the following: Slander, Libel, False accusations. 
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above.
            You will identify Defamation risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Defamation risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results

def pollution_information_ecosystem_societal_risk_agent(use_case_summary):
    prompt = PromptTemplate(
        template="""
            You are a risk agent focusing on Pollution of Information Ecosystem. 
            You will identify risks relating to potential pollution of the information ecosystem, potentially leading to societal harm. 
            Some of the sample types of risks you would consider include the following: Misinformation, Disinformation, Information overload.
            You MUST Evaluate if Risk Exists For Each of the sample types of risk referred above. 
            You will identify Pollution of Information Ecosystem risks in the use case below based on information provided.
            If the risk does not exist, please say that you do not perceive any Pollution of Information Ecosystem risk in the use case.
        \n{format_instructions}\n{query}\n""",
        input_variables=["query"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )

    chain = prompt | chat_llm | parser

    agent_results=chain.invoke({"query": use_case_summary})
    return agent_results


