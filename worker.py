import os
import torch
from langchain import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma
from langchain.llms import HuggingFaceHub
from ibm_watson_machine_learning.foundation_models.extensions.langchain import WatsonxLLM
from ibm_watson_machine_learning.foundation_models.utils.enums import ModelTypes, DecodingMethods
from ibm_watson_machine_learning.metanames import GenTextParamsMetaNames as GenParams
from ibm_watson_machine_learning.foundation_models import Model

# Check for GPU availability and set the appropriate device for computation.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Some of the code are from Coursera Lab https://www.coursera.org/learn/building-gen-ai-powered-applications/
# Modified and tested by Wenting Zheng

# Global variables
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None

Watsonx_API = "Your WatsonX API"
Project_id= "Your Project ID"

# Function to initialize llm and embeddings
def init_llm():
    global llm_hub, embeddings
    
    my_credentials = {
        "url"    : "https://us-south.ml.cloud.ibm.com"
    }
    params = {
            GenParams.MAX_NEW_TOKENS: 800, # The maximum number of tokens that the model can generate in a single run.
            GenParams.TEMPERATURE: 0.1,   # A parameter that controls the randomness of the token generation. A lower value makes the generation more deterministic, while a higher value introduces more randomness.
        }
    LLAMA2_model = Model(
            model_id= 'meta-llama/llama-2-70b-chat', 
            credentials=my_credentials,
            params=params,
            project_id="skills-network",  
        )
    llm_hub = WatsonxLLM(LLAMA2_model)  

    #Initialize embeddings using a pre-trained model to represent the text data.
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": DEVICE}
    )

# Function to process the uploaded document
def process_document(document_path, file_extension):
    global conversation_retrieval_chain
    # Get the file extension

    # Load the document based on the file type
    if file_extension == '.pdf':
        loader = PyPDFLoader(document_path)
    elif file_extension == '.doc' or file_extension == '.docx':
        loader = Docx2txtLoader(document_path)
    elif file_extension == '.txt':
        loader = TextLoader(document_path)
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

    documents = loader.load()

    # Split pdf into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64) # ---> use Recursive Character TextSplitter and specify the input parameters <---
    texts = text_splitter.split_documents(documents)
    
    # Create an embeddings database using Chroma from the split text chunks.
    db = Chroma.from_documents(texts, embedding=embeddings)
    
    # Build the QA chain, which utilizes the LLM and retriever for answering questions.
    conversation_retrieval_chain = RetrievalQA.from_chain_type(
        llm=llm_hub,
        chain_type="stuff",
        retriever= db.as_retriever(search_type="mmr", 
        search_kwargs={'k': 6, 'lambda_mult': 0.25}),
        return_source_documents=False
    )


# Function to process a user prompt
def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history

    # Pass the prompt and the chat history to the conversation_retrieval_chain object
    output = conversation_retrieval_chain({"query": prompt, "chat_history": chat_history})
    answer =  output["result"]
    
    # Update the chat history
    chat_history.append((prompt, answer))
    
    # Return the model's response
    return answer
    

# Initialize the language model
init_llm()
