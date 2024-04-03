import os
import langchain
from textwrap import dedent
import pandas as pd

from langchain_openai import ChatOpenAI
from langchain.vectorstores.faiss import FAISS
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain

from langchain_community.llms.ctransformers import CTransformers
from langchain_community.llms.ollama import Ollama
from langchain.llms.llamacpp import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import (
    ChatPromptTemplate
)
import openai

from rest_api_files.vectordbsearch_tools import VectorSearchTools_chroma



#for linux or mac with litellm and ollama
lite_llm = ChatOpenAI(
    openai_api_base="http://0.0.0.0:8000",
    model = "llama",
    api_key= "h",
    temperature=0.1,
    
)



def load_local_model(model_path):
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
    n_gpu_layers = 10  # Increase based on your GPU capacity. 0 means no gpu usage.
    n_batch = 2048  # Should be between 1 and n_ctx, consider the amount of RAM.
 
    llm_llama = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=n_gpu_layers,
        temperature=0.1,
        n_batch=n_batch,
        n_ctx=8000,
        max_tokens= 1024,
        top_k =5,
        f16_kv=True,
        callback_manager=callback_manager,
        verbose=True,
    )
    return llm_llama
       

ollama_llm = Ollama(model="mistral", num_ctx=4096,temperature=0.1, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])) 
#install Ollama application and run with 'ollama run 'model name''. for example: ollama run llama2


os.environ["OPENAI_API_KEY"] = " " #OpenAI API Key for gpt3.5 or gpt4 use

openai_llm = ChatOpenAI(temperature=0.1)


#model_path="./models/zephyr-7b-alpha.Q4_K_M.gguf"  #local model file. add model file (.gguf) in model models folder and replace the name/path

#local_llm = load_local_model(model_path)

llm=ollama_llm  #you can use whichever model you want (e.g ollama_llm)






#for RAG fusion
def generate_queries_chatgpt(original_query, memory):
    
    

    messages = (f"You are a helpful assistant that generates stand alone question based on the new user query and conversation history.\n\n  new user query: {original_query}.\n\n conversation history: {memory}. \n\n OUTPUT (stand alone question):"
    )
    

    response=llm.predict(messages)
    #generated_queries = response.choices[0]["message"]["content"].strip().split("\n")
    return response
  



class RAGbot:
      
    
    def run(prompt, memory): #need to pass user query and memory variable.
       
        #try:         
          
          # Get user input -> Generate the answer
          greetings = ['Hey', 'Hello', 'hi', 'hallo', 'hu' 'hello', 'hey', 'helloo', 'hellooo', 'g morning', 'gmorning', 'good morning', 'morning',
                      'good day', 'good afternoon', 'good evening', 'greetings', 'greeting', 'good to see you', 'guten morgen', 'morgen','guten tag', 'guten tag', 'guten abend', 'grüße', 'gruß', 'schön', 'sie zu sehen','its good seeing you', 'how are you', "how're you", 'how are you doing', "how ya doin'", 'how ya doin',
                      'how is everything', 'how is everything going', "how's everything going", 'how is you', "how's you"]
          
          compliment = ['thank you', 'thanks', 'thank' 'thanks a lot', 'thanks a bunch', 'great', 'ok', 'ok thanks', 'okay', 'great', 'good' 'awesome', 'nice', 'danke', 'danke', 'vielen dank', 'vielen dank', 'großartig', 'großartig', 'gut', 'fantastisch', 'hübsch']
                     
          prompt_template =dedent(r"""
              Verwenden Sie die folgenden Kontextelemente, um die Frage am Ende zu beantworten. Beantworten Sie die Frage in derselben Sprache.
            
              Antworten Sie nicht anhand Ihrer Trainingsdaten.
              Wenn die Antwort nicht im Kontext gefunden wird. Erfinden Sie keine Antwort. Erfinden Sie keine hypothetischen Antworten. Beantworten Sie nur die richtigen Informationen.
              Antworten Sie richtig und auf den Punkt. Fügen Sie keine unerwünschten Informationen hinzu.
              
              Stellen Sie sicher, dass Ihre Antworten in direktem Zusammenhang mit der Benutzerfrage und dem Chatverlauf stehen.
              
              Kontext:
              ---------
              {context}
              
              ---------

              Chatverlauf: 
              ---------
              {chat_history}
              ---------

              Question: 
              {question}

              Hilfreiche Antwort:
              """)
              
              

          PROMPT = PromptTemplate(
                  template=prompt_template, input_variables=[ "context", "question", "memory"]
              )

            
          chain = load_qa_chain(llm=llm, verbose= True, prompt = PROMPT, chain_type="stuff")
                  
        
        
          if prompt.lower() in greetings:
            response = 'Hallo, wie geht es dir? Ich bin hier, um Ihnen dabei zu helfen, Informationen aus Ihrer Akte zu erhalten. Wie kann ich Ihnen helfen?'
            
              
            return response
              
          elif prompt.lower() in compliment:
            response = 'Freut mich! Wenn Sie weitere Fragen haben, können Sie diese gerne stellen.'
            
              
            return response
            #memory.save_context({"question": prompt}, {"output": response})
              
          else:
            
            docs =VectorSearchTools_chroma.dbsearch(prompt)
            
            
            
            response = chain.run(input_documents=docs, question=prompt)
            
          
           
          
            
              
            return response
        
        #except Exception as e:
            
         #   "Sorry, the question is irrelevant or the bot crashed"
    


if __name__ == "__main__":
      print("## Welcome to the RAG chatbot")
      print('-------------------------------')
      
      while True:
        query = input("You: ")
        memory= None
        if query.lower() == 'exit':
            break
        chatbot= RAGbot
        result = chatbot.run(query)
        print("Bot:", result)