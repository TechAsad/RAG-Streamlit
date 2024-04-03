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
from langchain_community.chat_models import ChatGooglePalm
from langchain.llms.llamacpp import LlamaCpp
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import (
    ChatPromptTemplate
)
import openai
from googlesearch_tool import fetch_top_search_results
from rest_api_files.vectordbsearch_tools import VectorSearchTools_chroma
from langchain_community.vectorstores.chroma import Chroma
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings
)

     
ollama_llm = Ollama(model="llama2", num_ctx=16384, temperature=0.01, callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])) 
#install Ollama application and run with 'ollama run 'model name''. for example: ollama run llama2, ollama run solar



llm=ollama_llm #you can use whichever model you want (e.g ollama_llm, openai_llm or local_llm)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

#for RAG fusion
def generate_question(original_query, memory):
    
    model = llm

    messages = (f"""Create a SINGLE standalone question for good google search. The stand alone question should be based on the New question plus the Chat history. 
    If the New question can stand on its own you should return the New question. Do not do any reasoning or extra words. New question: \"{original_query}\", Chat history: \"{memory}\".  STAND ALONE QUESTION:"""
    )
    

    response=model.invoke(messages)
    #generated_queries = response.choices[0]["message"]["content"].strip().split("\n")
    return response


class internet_RAGbot:
      
    
    def run(prompt, memory): #need to pass user query and memory variable.
          
        try:  
               
          
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
            Answer truthfully, helpfully like a friend.
            Ensure that your answers are directly related to the user's query and chat history.
           
            context:
            
            
            {context}
            
        
            ---------
            
            Google Search Results:
            
            {results}

            chat history: 
            ---------
            {chat_history}
            ---------

            Question: 
            {question}

            Helpful Answer: 
            """)
              
              

          PROMPT = PromptTemplate(
                  template=prompt_template, input_variables=[ "context","results", "question", "chat_history"]
              )

            
          chain = load_qa_chain(llm=llm, verbose= True, prompt = PROMPT,memory=memory, chain_type="stuff")
                  
        
        
          if prompt.lower() in greetings:
            response = 'Hi, how are you? I am here to help you get information from your file. How can I assist you?'
            
              
            return response
              
          elif prompt.lower() in compliment:
            response = 'My pleasure! If you have any more questions, feel free to ask.'
            
              
            return response
          else:
            
            results=[]
            stand_alone = generate_question(prompt, memory)
            net_search= fetch_top_search_results(stand_alone)
            
        
                
            for i, result in enumerate(net_search, start=1):
              result= f"{i}. {result}\n"
              results.append(result)
              
            embeddings = SentenceTransformerEmbeddings(model_name="intfloat/multilingual-e5-large",model_kwargs = {'device': 'cpu'})
            db = Chroma(embedding_function= embeddings, persist_directory="./chroma_db")

            docs =db.similarity_search(stand_alone, 2)
            
            print(stand_alone)   
            print(net_search)
            response = chain.invoke(input={"input_documents":docs, "question":prompt, "results": results})
            
        
            
                
              
            return response["output_text"]
        
        except Exception as e:
            
            "Sorry, the question is irrelevant or the bot crashed"
    


if __name__ == "__main__":
      print("## Welcome to the RAG chatbot")
      print('-------------------------------')
      
      while True:
        query = input("You: ")
        if query.lower() == 'exit':
            break
        chatbot= internet_RAGbot
        result = chatbot.run(query)
        print("Bot:", result)