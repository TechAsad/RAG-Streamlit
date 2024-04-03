# Chatbot Application Readme



## Please add your files [pdf, docx] in 'files' folder for pre loading documents.


## Running the Application

Follow these steps to set up and run the entire application:

1. **Create a Virtual Environment:**
   
   python3 -m venv myenv
 

2. **Activate the Virtual Environment:**
   - On Windows:
    
     .\myenv\Scripts\activate
     
   - On Linux/Mac:
    
     source myenv/bin/activate
    

## if you want to use local llm model


3. install ollama (https://ollama.com/download) for your system

run ' ollama run llama2' in the terminal



4. **Install Required Modules:**
  
   pip install -r requirements.txt

5. **Ingest Data:**

This will imbed the data and create vector database from your files in `files` folder. only for preloading documents.
- **How to Run:** run it with `python data_ingest.py`  or `python3 data_ingest.py` in the terminal.


6. **Run Streamlit App:**
   - Start the Streamlit app with `streamlit run app.py` to use a graphical interface for the chatbot.

Note: Ensure that each step is executed in order for the proper functioning of the application.



## Rename `Streamlit.streamlit` to `.streamlit` for app Theme. 



## files

### 1. `data_ingest.py`
- **Purpose:** This file will imbed the data and create vector database from your files in `files` folder. Add unversity related documetns in files filder.
- **How to Run:** run it with `python data_ingest.py`  or `python3 data_ingest.py` in the terminal.


### 2. `main.py`
- **Purpose:** This file contains the main function of the chatbot. This can be used in frontend application for making call to chatbot.



### 2. `restapi_app.py`
- **Purpose:** This file is the RestAPI app. This can be used in frontend application in javascript for making call to chatbot.
- **How to Run:** Run it with `python restapi_app.py`  or `python3 restapi_app.py` in the terminal. It will put the app on .`http://127.0.0.1:5000/chatbot`


### 3. `app.py`
- **Purpose:** This file contains a Streamlit application for a user-friendly interface to interact with the chatbot.
- **How to Run:** Start the Streamlit app by running `streamlit run app.py` in the terminal.





