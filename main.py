import os
import langchain
from textwrap import dedent
import pandas as pd
import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain

from langchain_community.llms.ctransformers import CTransformers
from langchain_community.llms.ollama import Ollama
from langchain_community.chat_models import ChatGooglePalm
from langchain.llms.llamacpp import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import (
    ChatPromptTemplate
)
import openai

def select_model(llm_model):  

  ollama_llm = Ollama(model=llm_model, num_ctx=16384, temperature=0.1, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])) 
  #install Ollama application and run with 'ollama run 'model name''. for example: ollama run llama2, ollama run solar
  return ollama_llm

google_api_key = st.secrets["GOOGLE_API_KEY"]
#api_key2 = st.secrets["OPENAI_API_KEY"]
os.environ["GOOGLE_API_KEY"] = google_api_key

llm = ChatGooglePalm(temperature=0.1)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#for RAG fusion
def generate_question(original_query, memory, llm_model):
    
    model =llm

    messages = (f"""Create a SINGLE standalone question. your output should include only standalone question. The question should be based on the New question plus the Chat history. 
    If the New question can stand on its own you should return the New question. Do not do any reasoning or extra words. New question: \"{original_query}\", Chat history: \"{memory}\"."""
    )
    

    response=model.invoke(messages)
    #generated_queries = response.choices[0]["message"]["content"].strip().split("\n")
    return response



class RAGbot:
      
    
    def run(prompt, memory, db): #need to pass user query and memory variable.
        
          
        
        #try:  
               
          
          # Get user input -> Generate the answer
          greetings = ['Hey', 'Hello', 'hi', 'hu' 'hello', 'hey', 'helloo', 'hellooo', 'g morning', 'gmorning', 'good morning', 'morning',
                      'good day', 'good afternoon', 'good evening', 'greetings', 'greeting', 'good to see you',
                      'its good seeing you', 'how are you', "how're you", 'how are you doing', "how ya doin'", 'how ya doin',
                      'how is everything', 'how is everything going', "how's everything going", 'how is you', "how's you",
                      'how are things', "how're things", 'how is it going', "how's it going", "how's it goin'", "how's it goin",
                      'how is life been treating you', "how's life been treating you", 'how have you been', "how've you been",
                      'what is up', "what's up", 'what is cracking', "what's cracking", 'what is good', "what's good",
                      'what is happening', "what's happening", 'what is new', "what's new", 'what is neww', "g’day", 'howdy' 'ji']
          compliment = ['thank you', 'thanks', 'thank' 'thanks a lot', 'thanks a bunch', 'great', 'ok', 'ok thanks', 'okay', 'great', 'good' 'awesome', 'nice']
                      
          prompt_template =dedent(r"""
            Use the following pieces of context to answer the question at the end. Answer in the same language of the question.
            You are a helpful assitant to help user with question. 
            Answer the QUESTION truthfully and to the point.
            
            Ensure that your answers are directly related to the user's query and chat history.
           
            context:
            
            
            {context}
            
            ---------

            chat history: 
            ---------
            {chat_history}
            ---------

            QUESTION: 
            {question}

            ANSWER: 
            """)
              
              

          PROMPT = PromptTemplate(
                  template=prompt_template, input_variables=[ "context","question", "chat_history"]
              )

            
          chain = load_qa_chain(llm=llm, verbose= True, prompt = PROMPT,memory=memory, chain_type="stuff")
                  
        
        
          if prompt.lower() in greetings:
            response = 'Hi, how are you? I am here to help you get information from your file. How can I assist you?'
            
              
            return response
              
          elif prompt.lower() in compliment:
            response = 'My pleasure! If you have any more questions, feel free to ask.'
            
              
            return response
          else:
            
            if db is not None:
              #stand_alone = generate_question(prompt, memory, llm_model)
              
              docs =db.similarity_search(prompt, 6)
              print(docs)
              print(prompt)   
              
              response = chain.invoke(input={"input_documents":docs, "question":prompt})
              return response["output_text"]
            else:
                 
                prompt_chat = ChatPromptTemplate.from_template("you are a helpful assistant. Answer the QUESTION truthfully and to the point. \n\n Chat History: {chat_history} \n\n QUESTION: {question}  \n\n OUTPUT :")
                chain2 = prompt_chat | llm
                response = chain2.invoke({"chat_history": memory, "question": prompt}).content
                
          
              
                
              
                return response
        
        #except Exception as e:
            
         #   "Sorry, the question is irrelevant or the bot crashed"
    


if __name__ == "__main__":
      print("## Welcome to the RAG chatbot")
      print('-------------------------------')
      
      while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        chatbot= RAGbot
        result = chatbot.run(query)
        print("Bot:", result)