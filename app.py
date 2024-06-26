import os

import streamlit as st
from PyPDF2 import PdfReader
import langchain
from textwrap import dedent
import json

from langchain_community.callbacks import StreamlitCallbackHandler
from langchain.memory import ConversationBufferMemory
import subprocess
from main import RAGbot
from datetime import date

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders.word_document import UnstructuredWordDocumentLoader
import tempfile
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.document_loaders.excel import UnstructuredExcelLoader
from langchain.document_loaders.pdf import PyPDFLoader
from langchain.document_loaders.json_loader import JSONLoader
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain.vectorstores.faiss import FAISS

from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings



import hmac
st.set_page_config(page_title='Personal AI Chatbot', page_icon='books')


st.markdown(
    """
    <style>
        [data-testid=stImage]{
            text-align: center;
            display: block;
            margin-top: -10%;
            margin-left: 35%;
            margin-right:0%;
            width: 200px;
            height: 80px;
            
            
    }
    
    </style>
    """, unsafe_allow_html=True
)

st.image('Logo.jpeg', width=200)


def check_password():
    """Returns `True` if the user had the correct password."""

    def password_entered():
        """Checks whether a password entered by the user is correct."""
        if hmac.compare_digest(st.session_state["password"], st.secrets["password"]):
            st.session_state["password_correct"] = True
            del st.session_state["password"]  # Don't store the password.
        else:
            st.session_state["password_correct"] = False

    # Return True if the password is validated.
    if st.session_state.get("password_correct", False):
        return True

    # Show input for password.
    st.subheader("Login Page")
    st.text_input(
        "Password", type="password", on_change=password_entered, key="password"
    )
    if "password_correct" in st.session_state:
        st.error("😕 Password incorrect")
    return False


if not check_password():
    st.stop()  # Do not continue if check_password is not True.


#st.markdown("<h3 style='text-align: center; color: black;'> AI Assistant </h3>", unsafe_allow_html=True)



st.markdown(
    """
    <style>
    [data-testid="stChatMessageContent"] p{
        font-size: 0.85rem;
        color: #000000

    }
    </style>
    """, unsafe_allow_html=True
)



import glob

def load_files_from_folder(folder_path):
    # Check if the folder path exists
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return []

    # Use glob to find all files in the folder
    files = glob.glob(os.path.join(folder_path, "*"))
    
    # Return the list of files
    return files

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

    def __str__(self):
        page_content_str = self.page_content
        metadata_str =self.metadata
        return f"[Document(page_content={page_content_str}, metadata={metadata_str})]"


@st.cache_resource(show_spinner=False)
def process_pdf_docx(path):
    try:
        with st.spinner(text=f"Embedding Your Files from '{path}' "):
        
            files = load_files_from_folder(path)
            # Read text from the uploaded PDF file
            data = []
            for file in files:
                
            
                if file.lower().endswith(".pdf"):

                    
                    loader = PyPDFLoader(file_path=file)
                    documents = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
                    data += text_splitter.split_documents(documents)


                if file.lower().endswith(".csv"):
                    
                
                    loader = CSVLoader(file_path=file, encoding="utf-8", csv_args={
                                'delimiter': ','})
                    documents = loader.load()
                    
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        
                    data += text_splitter.split_documents(documents)
                    
            
                
                if file.lower().endswith(".json"):
                    
                
                    loader = JSONLoader(
                    file_path=file,
                    jq_schema='.',
                    text_content=False)
                    documents = loader.load()
                    print(documents)
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
        
                    data += text_splitter.split_documents(documents)
                    
                if file.lower().endswith(".docx"):

                
                    loader = UnstructuredWordDocumentLoader(file_path=file)
                    documents = loader.load()
                    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

                    data += text_splitter.split_documents(documents)
                

            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    

            vectorstore = FAISS.from_documents(data, embeddings)
            #vectorstore.save_local("./faiss")
            return vectorstore
    except Exception as e:
        
            
        st.warning(f"Sorry, the path '{path}' does not exist.")


@st.cache_resource(show_spinner=False)
def process_uploaded_files(uploaded_file):
    try:
        with st.spinner(text="Embedding Your Files"):

            # Read text from the uploaded PDF file
            data = []
            for file in uploaded_file:
                split_tup = os.path.splitext(file.name)
                file_extension = split_tup[1]
            
                if file_extension == ".pdf":

                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file1:
                        tmp_file1.write(file.getvalue())
                        tmp_file_path1 = tmp_file1.name
                        loader = PyPDFLoader(file_path=tmp_file_path1)
                        documents = loader.load()
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
                        data += text_splitter.split_documents(documents)


                if file_extension == ".csv":
                    
                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(file.getvalue())
                        tmp_file_path = tmp_file.name

                        loader = CSVLoader(file_path=tmp_file_path, encoding="utf-8", csv_args={
                                    'delimiter': ','})
                        documents = loader.load()
                        
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)
            
                        data += text_splitter.split_documents(documents)
                        
                        
                
                if file_extension == ".docx":

                    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
                        tmp_file.write(file.getvalue())
                        tmp_file_path = tmp_file.name
                        loader = UnstructuredWordDocumentLoader(file_path=tmp_file_path)
                        documents = loader.load()
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=50)

                        data += text_splitter.split_documents(documents)
                    

            
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    

            # Create a FAISS index from texts and embeddings

            vectorstore = FAISS.from_documents(data, embeddings)
            #vectorstore.save_local("./faiss")
            return vectorstore
    except Exception as e:
        
            
        st.warning(f"Sorry, the path '{path}' does not exist.")




def get_llama_names():
    # Run the 'ollama list' command and capture the output
    output = subprocess.run(['ollama', 'list'], capture_output=True, text=True)
    
    # Split the output into lines and extract the names
    names = []
    for line in output.stdout.split('\n'):
        if line.strip().startswith('NAME'):
            continue
        if line.strip() == '':
            continue
        name = line.split()[0]
        names.append(name)
    
    return names

# Get the list of names
#llm_names = get_llama_names()

with st.sidebar:
    uploaded_file =  st.file_uploader("Upload your files",
    help="Multiple Files are Supported",
    type=['pdf', 'docx', 'csv'], accept_multiple_files= True)


with st.sidebar:
 
    #st.sidebar.subheader("Select LLM")
    
    #selected_llm = st.selectbox(
    #"Available LLMs",
    #llm_names,
    #index=None,
    #placeholder="Select..."
#)   
    #if selected_llm is not None:
    #    st.success(f"Selected LLM: {selected_llm}")
    #else:
    #    st.warning("Please Select the LLM")
    path =  st.text_input("Enter RAG Directory", placeholder="Enter")




#llm_model = selected_llm


if 'history' not in st.session_state:  
        st.session_state['history'] = []


if "messages" not in st.session_state or st.sidebar.button("Clear Chat"):
    st.session_state["messages"]= []


if path:
    db = process_pdf_docx(path)
    files = load_files_from_folder(path)
    total_files= len(files)
    st.sidebar.subheader(f"Embedded Files form '{path}'")
    for file_path in files:
        file_name = os.path.basename(file_path)
        st.sidebar.write(file_name)
    if len(files)>1:
        st.success(f"{total_files} files are embedded from {path}")
        
    else: 
        st.success(f"{total_files} file is embedded from {path}")
else:
    
    db = None


if uploaded_file:
    db2= process_uploaded_files(uploaded_file)
    
    total_files= len(uploaded_file)
    if total_files>1:
        st.success(f"{total_files} files are embedded from uploaded files")
    else: 
        st.success(f"{total_files} file is embedded from uploaded file") 
else:
    
    db2 = None

if uploaded_file and path is None:
    st.warning("Please Provide Directory Path for RAG")

def main():
    
    try:
           
        memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", human_prefix= "User", ai_prefix= "Assistant")
        for msg in st.session_state.messages:
            
            if msg["role"]== 'Assistant':
                st.chat_message(msg["role"], avatar="logo_bot.png").write(msg["content"])
            else: 
                st.chat_message(msg["role"], avatar = "user.png").write(msg["content"])
                
                
        #prompt = prompt
        if prompt := st.chat_input(placeholder="Type your question!"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            for i in range(0, len(st.session_state.messages), 2):
                if i + 1 < len(st.session_state.messages):
                    user_prompt = st.session_state.messages[i]
                    ai_res = st.session_state.messages[i + 1]
                    
                
                    user_content = user_prompt["content"]
                    
                    
                    ai_content = ai_res["content"]
                    
                    # Concatenate role and content for context and output
                    user = f"{user_content}"
                    ai = f"{ai_content}"
                    
                    memory.save_context({"question": user}, {"output": ai})
        
            
            st.chat_message("user", avatar = "user.png").write(prompt)
             
          
            with st.chat_message("Assistant", avatar= "logo_bot.png"):
                with st.spinner('Assistant...'):
                    chatbot= RAGbot
                    
                    response = chatbot.run(prompt, memory, db, db2)
                   
                        
                    st.session_state.messages.append({"role": "Assistant", "content": response})
                    st.write(response) 
            #memory.save_context({"question": prompt}, {"output": response})
                           
                                          
    except Exception as e:
        with st.chat_message("Assistant", avatar= "logo_bot.png"):
            
            st.warning(f"Sorry, the bot crashed or the folder '{path}' does not exist.")
 



hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 


if __name__ == '__main__':
    main()





