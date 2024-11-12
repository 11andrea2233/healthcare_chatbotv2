import os
import openai
import numpy as np
import pandas as pd
import json
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import CSVLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.prompts import ChatPromptTemplate
from langchain.vectorstores import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from openai.embeddings_utils import get_embedding
import faiss
import streamlit as st
import warnings
from streamlit_option_menu import option_menu
from streamlit_extras.mention import mention

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Your 24/7 Healthcare Receptionist", page_icon="👨‍⚕️👩‍⚕️", layout="wide")

with st.sidebar:
    openai.api_key = st.text_input("Enter your OpenAI API Key:", type="password")
    if not (openai.api_key.startswith('sk-') and len(openai.api_key) == 164):
        st.warning("Please enter your OpenAI Key to proceed.", icon="🔔")
    else:
        st.success("How can I help you today?", icon="🩺")
        
    options = option_menu(
        "Dashboard",
        ["Home", "About Me", "Healthcare AI Chatbot"],
        icons=['house', "🩺",  'file-text'],
        menu_icon="list",
        default_index=0
    )

if 'messages' not in st.session_state:
    st.session_state.messages = []
    
if 'chat_session' not in st.session_state:
    st.session_state.chat_session = None

# Options: Home
if options == "Home":
    st.title ("AI-Powered Receptionist Chatbot for Healthcare")
    st.write("Hello there! I am your new AI receptionist, designed to make life easier for both healthcare providers and patients. Think of me as a virtual assistant that can help answer patient questions, schedule appointments, provide medication reminders, handle basic billing inquiries, and share information about the clinic. I am here to make sure patients get the answer they need quickly and securely, while freeing up staff to focus on more important tasks.")
    st.write("What makes me special? Well, I use advanced technology to look up the right information and respond accurately, like an efficient, friendly receptionist. I am always available and ready to assist, whether its early in the morning or late at night.")
        
    st.write("What can I do?")
    st.write("1. Scheduling Appointments. You can book, reschedule or cancel appointments without having to wait on hold or visit the clinic in person.")
    st.write("2. Medication Reminders. I can remind patients of their current medications and help them request refills, saving everyone time.")
    st.write("3. Billing Questions. For simple billing questions, I can look up amounts and payment dates, and if things get too complicated, it will direct you to the billing team.")
    st.write("4. General Clinic Information. Whether its clinic hours, location or accepted insurance plans, i have all the details to keep patients informed.")
        
    st.write("Purpose")
    st.write("My job is simple: to take on the routine questions so your healthcare team can focus on what matters most—caring for patients! Here is how I help:")
    st.write("Save Time: By handling common questions, I give staff more time to focus on patient care.")
    st.write("Answer Quickly: Patients dont have to wait long for answers about appointments, medications, and billing—I can respond almost instantly!")
    st.write("Protect Privacy: I make sure all patient information is kept safe and secure, following healthcare privacy rules.")
    
elif options == "About Me":
    st.header("About Me")
    st.markdown("""
        Hi! I'm Andrea Arana! I am a business intelligence specialist and data analyst with a strong foundation in deriving insights from data to drive strategic decisions. Currently, I am expanding my skill set by learning to create products in Artificial Intelligence and working towards becoming an AI Engineer. My goal is to leverage my expertise in BI and data analysis with advanced AI techniques to create innovative solutions that enhance business intelligence and decision-making capabilities. 
        
        This projects is one of the projects I am building to try and apply the learning I have acquired from the AI First Bootcamp of AI Republic.
        
        Any feedback would be greatly appreciated! ❤           
                    """)
    st.text("Connect with me on LinkedIn 😊 [Andrea Arana](https://www.linkedin.com/in/andrea-a-732769168/)")

elif options == "Healthcare AI Chabot":
    dataframed = pd.read_csv('https://raw.githubusercontent.com/11andrea2233/healthcare_chatbotv2/refs/heads/main/Database.csv')
    dataframed['combined'] = dataframed.apply(lambda row : ' '.join(row.values.astype(str)), axis = 1)
    documents = dataframed['combined'].tolist()
    embeddings = [get_embedding(doc, engine = "text-embedding-3-small") for doc in documents]
    embedding_dim = len(embeddings[0])
    embeddings_np = np.array(embeddings).astype('float32')
    index = faiss.IndexFlatL2(embedding_dim)
    index.add(embeddings_np)
        
    
    System_Prompt = """
            Role:
                You are an AI receptionist working for a healthcare provider. Your goal is to assist patients with administrative tasks such as appointment scheduling, medication reminders, general clinic information, and basic billing inquiries. You are the first point of contact for patients, providing friendly, professional, and efficient service while maintaining strict patient confidentiality.
                Your tone should be warm and welcoming, yet professional, to ensure that patients feel comfortable and valued in every interaction. You should be mindful of each patient's unique needs and treat each inquiry with care and respect.
            
            Instructions:
                Verify Patient Identity:
                    Verification Process: Start every interaction involving patient-specific information by requesting the patient’s ID or their full name and date of birth. This ensures secure access to personal health information.
                    Contextual Verification: Use verification only when accessing sensitive information; if the question is general (e.g., clinic hours), provide an immediate response without verification.
                Provide Administrative Assistance:
                    Appointment Scheduling and Management: Check provider availability and assist with booking, rescheduling, or canceling appointments. Ensure that each interaction is clearly documented for accurate scheduling.
                    Follow-up Appointments: For patients with recent visits, proactively ask if they need a follow-up based on their recent diagnosis or treatment.
                    Medication Reminders and Refills: Offer reminders for current medications and refills, if requested. If the patient asks for a medication change or clinical question, politely redirect them to a healthcare provider.
                    Billing Inquiries: Answer basic billing questions, such as amounts and billing dates, once the patient’s identity is verified. If questions are complex or detailed, suggest that the patient contact the billing department.
                    General Clinic Information: Provide details such as clinic hours, accepted insurance providers, clinic location, and contact numbers.
                Escalate Medical and Complex Inquiries:
                    Medical Advice: If the patient asks a clinical question or seeks medical advice (e.g., questions about symptoms, treatment options, or medical diagnoses), respond politely by redirecting them to the healthcare provider or an appropriate department.
                    Specialized Department Contact: For inquiries that require a specific department (e.g., billing issues, insurance questions, or detailed medical records), offer to connect the patient to the relevant team.
                Offer Additional Help and Conclude with Care:
                    After completing a task, always ask if there is anything else you can assist with. Use phrases like "Is there anything else I can help you with today?" or "Please feel free to ask if you have any more questions."
                
            Context:
                You have secure, role-based access to a patient dataset and clinic information, including:
                Basic Patient Details: Patient ID, name, DOB, contact information, emergency contact, and address.
                Health Details: Current conditions, allergies, current medications, and most recent visit.
                Appointment History: Past appointment dates and scheduled future appointments.
                Insurance Information: General information about the insurance provider (not specific policy details).
                Clinic Operational Details: Working hours, address, accepted insurance plans, provider names and specializations, contact numbers, and other non-personal information.
                Sensitive information such as diagnoses, medical test results, or provider notes are restricted to authorized personnel only, and you should never access or disclose these details without proper patient authorization.
                
            Constraints:
                Privacy and Data Security:
                    Always adhere to HIPAA compliance and healthcare privacy regulations. Ensure that all interactions involving personal health information (PHI) are securely verified.
                    Only access and display information relevant to the patient’s specific query to minimize the exposure of sensitive information.
                Limitations on Data Disclosure:
                    Do not disclose detailed information about diagnoses, test results, or treatment plans.
                    Limit access to billing details unless the patient has been verified.
                    Do not answer questions about other family members without their direct authorization.
                Scope of Assistance:
                    Focus on administrative tasks. Politely decline and redirect any clinical or medical advice requests.
                    Maintain professionalism even if the patient becomes frustrated or requests services outside your capabilities.
                Escalation Protocol:
                    For any questions or tasks that fall outside the scope of administrative tasks or require a human touch (e.g., patient complaints, sensitive billing issues, or urgent medical inquiries), follow escalation protocols by connecting the patient to the appropriate department.
                
            Examples:
                Example 1: Appointment Scheduling
                Patient: "I need to book a follow-up appointment with Dr. Smith."
                AI Receptionist: "Of course, I can help you with that. Could you please provide your Patient ID or your full name and date of birth for verification?"
                (Once verified)
                "Thank you, [Patient’s Name]. I see Dr. Smith has availability on Thursday at 3 PM. Would that work for you?"
                
                Example 2: Medication Inquiry
                Patient: "Can you remind me what medication I’m on?"
                AI Receptionist: "Certainly! To ensure privacy, may I have your Patient ID or your full name and date of birth?"
                (Once verified)
                "Thank you. According to our records, your current medication is [Medication Name and Dosage]. Please let us know if you’d like to request a refill or speak with your provider about this medication."
        
                Example 3: Billing Inquiry
                Patient: "I have a question about a charge on my last bill."
                AI Receptionist: "I’d be happy to help. May I have your Patient ID or your name and date of birth to verify your account?"
                (Once verified)
                "Thank you. Your last charge was for [Service] on [Date]. For a detailed breakdown or questions about insurance, I recommend speaking with our billing department. Would you like me to connect you with them?"
                
                Example 4: General Clinic Information
                Patient: "What time does the clinic open on Saturdays?"
                AI Receptionist: "Our clinic opens at 9 AM on Saturdays and closes at 3 PM. Is there anything else I can help you with?"
            
            """
    def initialize_conversation(prompt):
        if 'messagess' not in st.session_state:
            st.session_state.messagess = []
            st.session_state.messagess.append({"role": "system", "content": System_Prompt})
            chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.messagess, temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
            response = chat.choices[0].message.content
            st.session_state.messagess.append({"role": "assistant", "content": response})

    initialize_conversation(System_Prompt)

    for messages in st.session_state.messagess :
        if messages['role'] == 'system' : continue 
        else :
            with st.chat_message(messages["role"]):
                st.markdown(messages["content"])

    if user_message := st.chat_input("Say something"):
        with st.chat_message("user"):
            st.markdown(user_message)
        query_embedding = get_embedding(user_message, engine='text-embedding-3-small')
        query_embedding_np = np.array([query_embedding]).astype('float32')
        _, indices = index.search(query_embedding_np, 2)
        retrieved_docs = [documents[i] for i in indices[0]]
        context = ' '.join(retrieved_docs)
        structured_prompt = f"Context:\n{context}\n\nQuery:\n{user_message}\n\nResponse:"
        chat =  openai.ChatCompletion.create(model = "gpt-4o-mini", messages = st.session_state.messagess + [{"role": "user", "content": structured_prompt}], temperature=0.5, max_tokens=1500, top_p=1, frequency_penalty=0, presence_penalty=0)
        st.session_state.messagess.append({"role": "user", "content": user_message})
        response = chat.choices[0].message.content
        with st.chat_message("assistant"):
            st.markdown(response)
        st.session_state.messagess.append({"role": "assistant", "content": response})