import streamlit as st
# from llama_index import SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor, PromptHelper
from langchain import OpenAI
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.chains import ChatVectorDBChain
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chains import VectorDBQA
import sys
import uuid
import shutil
import os
# import pymongo
import streamlit as st
import base64
import time

import config
os.environ["OPENAI_API_KEY"] = config.OPENAI_API_KEY
OPENAI_API_KEY = config.OPENAI_API_KEY
import openai
openai.organization = config.OPENAI_ORG
openai.api_key = config.OPENAI_API_KEY
# pip install streamlit-chat  
from streamlit_chat import message

satisfaction_levels = {
    "😄": "Very satisfied",
    "🙂": "Satisfied",
    "😐": "Average",
    "😞": "Not satisfied",
    "Not Evaluated": "Not Evaluated"
}
# def ask_bot_prev(input_index = 'index.json', input_text = 'Hello'):
#   index = GPTSimpleVectorIndex.load_from_disk(input_index)
#   response = index.query(input_text, response_mode="compact")
#   return response.response



def ask_bot(input_text):
    print("searching DB index")
    persist_directory = st.session_state['db']
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    #print(persist_directory)
    docsearch = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
    #print(docsearch)
    # vectorstore = Chroma(
    #     collection_name="langchain_store",
    #     embedding_function=embeddings,
    #     persist_directory=persist_directory,
    # )
    #print(docsearch.similarity_search_with_score(query="Give the modified code for getBank method where the database should be used SQL not mongodb.", k=1))
    from langchain.prompts.chat import (
        ChatPromptTemplate,
        SystemMessagePromptTemplate,
        AIMessagePromptTemplate,
        HumanMessagePromptTemplate,
    )
    from langchain.schema import(
        AIMessage,
        HumanMessage,
        SystemMessage
    )
    system_template=""" Use the following pieces of context and chat history to answer the users question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    ---------------
    {context}
    """
    message =[
        SystemMessagePromptTemplate.from_template(system_template),
        HumanMessagePromptTemplate.from_template("{question}")
    ]
    prompt = ChatPromptTemplate.from_messages(message)
    chain_type_kwargs = {"prompt": prompt}
    search_kwargs = {"k": 1}
    #qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(), chain_type_kwargs=chain_type_kwargs, search_kwargs=search_kwargs)
    #qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever(search_kwargs={"k": 1}))
    #qa = RetrievalQA.from_chain_type(llm=OpenAI(), chain_type="stuff", retriever=docsearch.as_retriever())
    vectordbkwargs = {"search_distance": 0.9}
    chat_history = []
    try:

        if st.session_state['no_doc'] > 1:
            # st.write("document greater 1")
            qa = ChatVectorDBChain.from_llm(ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0), docsearch, qa_prompt=prompt)
            
            #result = qa.run(input_text)
            #print(result)
            result = qa({"question": input_text, "chat_history": chat_history})
        else:
            # st.write("document 1")
            qa = ChatVectorDBChain.from_llm(ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0), docsearch, qa_prompt=prompt, top_k_docs_for_context=1)
            
            #result = qa.run(input_text)
            #print(result)
            result = qa({"question": input_text, "chat_history": chat_history})
    except Exception as e:
        qa = ChatVectorDBChain.from_llm(ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0), docsearch, qa_prompt=prompt, top_k_docs_for_context=1)
        result = qa({"question": input_text, "chat_history": chat_history})

    
    #qa = ConversationalRetrievalChain.from_llm(OpenAI(temperature=0), docsearch.as_retriever(), return_source_documents=False, qa_prompt=prompt)
   
    # return "result"
    return result["answer"]
 

def initialize_robot(project_name):
    project_path = os.path.join("./", project_name)
    #project_path = "./source_codes"
    #print(project_path)
    loader = DirectoryLoader(project_path, glob="**/*.txt")
    documents = loader.load()
    # print(documents)
    st.write("Number of document to be processed: %d" % len(documents))
    st.session_state['no_doc'] = len(documents)
    text_splitter = CharacterTextSplitter(chunk_overlap=0, chunk_size=1500)
    texts = text_splitter.split_documents(documents)
   
    # print("----------------Now Textssss---------------")
    # print(texts)
    persist_directory = os.path.join(project_path,"db")
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    st.session_state['db'] = persist_directory
    if os.path.exists(persist_directory):
        # print("----------------Removing all documents---------------")
        shutil.rmtree(persist_directory)
        print( persist_directory)

    if not os.path.exists(persist_directory):
        docsearch = Chroma.from_documents(texts, embeddings, persist_directory=persist_directory)
        docsearch.persist()
        
    



def generate_response(prompt):
   
   message = ask_bot(prompt)
   return message

#Creating the chatbot interface
st.title("CoDescribe: The Intelligent Code Analyst")
with st.sidebar:
    st.title("Example Usecases")
    st.markdown("## Code Explanation\n" 
                "### _Prompts_\n"
                "- Explain the getSalary method used in the Employee Class (e.g. Java Project)\n"
                "- Explain the StoreDetails method of the in the Employee controller (e.g.  Node js Project)\n"
                "- Explain the sendMail method related to Notification task. (Normal type of query)\n"
                "## Documentation Generation\n"
                "### _Prompts_\n"
                "- Create a documentation for each method of the Employee class including the sample API calls for each method and respective request body.\n"
                "## Error Correction and Code Optimization\n"
                "### _Prompts_\n"
                "- Check the StoreDetails method and find if any error exists. If error found, correct the code and give the corrected code.\n" 
                "- Improve the StoreDetails method for example add exception handling if required.\n"
                "- Optimize the StoreDataCSV method.\n "
                "## Others\n"
                "### _Prompts_\n"
                "- Is there any data validation method used in the getUserData method? Where is the definition of this data validation method?.\n" 
                "- Give the modified code for storeData method where the database should be used SQL not mongodb.\n" 
                "- Optimize the StoreDataCSV method \n"
                "- Convert the storeData method to corresponding java code (suppose original storeData() is in python)"
                
                )

# Storing the chat
if 'generated' not in st.session_state:
    st.session_state['generated'] = []

if 'q_and_res' not in st.session_state:
    st.session_state.q_and_res = {
        "query": [],
        "response": []
    }
if 'no_doc' not in st.session_state:
    st.session_state['no_doc'] = ""

if 'project' not in st.session_state:
    st.session_state['project'] = ""
if 'db' not in st.session_state:
    st.session_state['db'] = ""

if 'past' not in st.session_state:
    st.session_state['past'] = []
if 'name' not in st.session_state:
    st.session_state['name'] = ""
if 'email' not in st.session_state:
    st.session_state['email'] = ""
if 'phone_number' not in st.session_state:
    st.session_state['phone_number'] = ""
if 'satisfaction_level' not in st.session_state:
    st.session_state['satisfaction_level'] = ""
if 'conversation' not in st.session_state:
    st.session_state['conversation'] = []
if 'user_input' not in st.session_state:
        st.session_state.user_input = ''
# Method for user feedback
def give_feedback(document):
    mongo_uri = "mongodb://localhost:27017/"
    # Create a MongoClient object
    client = pymongo.MongoClient(mongo_uri)
    # Access the database
    db = client.CoDescribe
    # Access the collection
    collection = db.user_feedback
    result = collection.insert_one(document)

    # Print the ID of the inserted document
    st.write("Thank You for giving feedback! We sincerely appreciate your feedback." )


# We will get the user's input by calling the get_text function
def get_text():
    
    st.session_state.user_input = st.session_state.widget
    st.session_state.widget = ''
    #st.text_input('Something', key='widget', on_change=submit)
    #input_text = st.text_input("You: ", key="input")
    #user_prompt = input_text
    # user_prompt = st.session_state.user_input
    # return user_prompt

# Section for Getting User Details and Project Creation
def validate_phone_number(phone):
    # Check if the phone number contains only digits
    if not phone.isdigit():
        return False
    else:
        return True
def create_folder(name, phone, email):
    st.session_state.name = name
    st.session_state.phone_number = phone
    st.session_state.email = email
    name = name.replace(" ", "")
    unique_id = str(uuid.uuid4()) # Generate a unique ID for the folder
    folder_name = f"{name}_{phone}_{unique_id}"
    os.mkdir(folder_name)
    st.session_state.project = folder_name
    st.session_state.name = folder_name
    return folder_name

st.title("Step 1: Create a New Project")
st.write("Enter your details below to create a new project. Do not forget to press ENTER or TAB after typing in the text boxes.")

name = st.text_input("Name")
email = st.text_input("Email")
phone = st.text_input("Phone Number")
project_name = ""
if st.button("Create Project"):
    if name and email and phone:
        if not validate_phone_number(phone):
            st.warning("Phone number should contain only digits.")
        else:
            folder_name = create_folder(name, phone, email)
            project_name = folder_name
            st.write("Project created successfully")
    else:
        st.warning("Please enter your name, email, and phone number to create a project.")



st.title("Step 2: Upload Files")
# File Upload
#uploaded_file = st.file_uploader("Choose a file")
if st.session_state.project == "":

    st.warning("Create Project First")
else:

    uploaded_files = st.file_uploader("Choose files", accept_multiple_files=True)
    if uploaded_files is not None:
        for uploaded_file in uploaded_files:
            try:
                # Decode the file's contents using UTF-8 encoding
                file_contents = uploaded_file.read().decode("utf-8")

                # Get the filename from the uploaded file object
                actual_filename = uploaded_file.name
                filename = uploaded_file.name+".txt"
                with open(os.path.join(st.session_state.project, filename), "w") as f:
                    f.write(file_contents)

                # Display a success message to the user
                st.write(f"File '{actual_filename}' saved successfully!")
            except Exception as e:
                st.warning("Error occurred while processing. Don't upload other than programming files.")



# Now User Chatting

st.title("Step 3: Chat with the Bot")
st.write("If you add new files please start the chatbot again")
if st.button("Start Chatbot"):
    if st.session_state.project == "":
        st.warning("Create Project First")
    else:
        initialize_robot(st.session_state.project)
enable_memory = st.checkbox("Enable memory")
#user_input = get_text()
st.text_input('You:', key='widget', on_change=get_text)
user_input = st.session_state.user_input
if enable_memory:
    if user_input:
        
        output = generate_response(user_input)
        # store the output 
        st.session_state.past.append(user_input)
        st.session_state.generated.append(output)
        st.session_state.q_and_res["query"].append(user_input)
        st.session_state.q_and_res["response"].append(output)

    if st.session_state['generated']:
        
        for i in range(len(st.session_state['generated'])-1, -1, -1):
            message(st.session_state["generated"][i], key=str(i))
            message(st.session_state['past'][i], is_user=True, key=str(i) + '_user')
        user_input = ""
        st.session_state.user_input = ""
else:
    if user_input:
        output = generate_response(user_input)
        st.session_state.q_and_res["query"].append(user_input)
        st.session_state.q_and_res["response"].append(output)
        message(output)
        message(user_input, is_user=True)

        user_input = ""
        st.session_state.user_input = ""


# User Feedback
# st.title("Step 4: Please Give Feedback")
# st.markdown("### _Rating for Code Explanation_\n")
# code_explanation_satisfaction = st.radio("Satisfaction For Code Explanation", list(satisfaction_levels.keys()))
# st.markdown("### _Rating for Documentation Generation_\n")
# documentation_satisfaction = st.radio("Satisfaction For Documentation Generation", list(satisfaction_levels.keys()))
# st.markdown("### _Rating for Error Checking and Code Optimization_\n")
# error_optimization_satisfaction = st.radio("Satisfaction for Error Checking and Code Optimization", list(satisfaction_levels.keys()))
# st.markdown("### _Rating for Other Chat Experiences_\n")
# other_chat_satisfaction = st.radio("Satisfaction for Other Chat Experiences", list(satisfaction_levels.keys()))
# if st.button("Send Feedback"):
#         if code_explanation_satisfaction and documentation_satisfaction and error_optimization_satisfaction:
#             documents_dict = [{"query": q, "response": r} for q, r in zip(st.session_state.q_and_res["query"], st.session_state.q_and_res["response"])]
#             document = {
#                 "name": st.session_state.name,
#                 "email": st.session_state.email,
#                 "phone_number": st.session_state.phone_number,
#                 "code_explanation_satisfaction_level": satisfaction_levels[code_explanation_satisfaction],
#                 "documentation_satisfaction_level": satisfaction_levels[documentation_satisfaction],
#                 "error_optimization_satisfaction_level": satisfaction_levels[error_optimization_satisfaction],
#                 "other_chat_satisfaction_level": satisfaction_levels[other_chat_satisfaction],
#                 "chat_history": documents_dict

#             }
#             give_feedback(document)
            
#         else:
            
#             st.warning("How Satisfied You are! Please select satisfaction level for each usecase")
