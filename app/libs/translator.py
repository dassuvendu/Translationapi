import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import TextLoader
from operator import itemgetter
import chromadb
from chromadb.config import Settings

import requests


load_dotenv()

class GPTLanguageTranslator:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        if not self.openai_api_key:
            raise ValueError("OpenAI API key is not set in the .env file.")
        
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

        self.chroma_client = chromadb.PersistentClient(path="./chromadb", settings=Settings(anonymized_telemetry=False))

    def translate_text(self, text, target_language):
        vectorstore = Chroma(client=self.chroma_client, embedding_function=OpenAIEmbeddings())
        retriever = vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": 4, 'fetch_k': 50})
    
        template = """I want you to act as a language translator. 
        I will provide you a text and a target language. 
        You will translate the text into the target language. 
        You will return only the translated text.

        You can use this context to improve your translation.
        Context: {context}
        Text: {text}
        Target Language: {target_language}
        Translated Text: """

        prompt = ChatPromptTemplate.from_template(template)

        model = ChatOpenAI(model_name="gpt-3.5-turbo-0125", temperature=0)

        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        chain = (
            { "target_language": RunnableLambda(lambda x: target_language), "text": RunnablePassthrough(), "context": retriever | format_docs}
            | prompt
            | model
            | StrOutputParser()
        )

        response = chain.invoke(text)

        return response

    async def train_model(self):
        url = "https://aitranslationhub.co/api/translation-records"  # Replace with your API endpoint URL

        # Make the GET request
        response = requests.get(url)
        print(response)
        # Check if the request was successful (status code 200)
        if response.status_code == 200:
            # Parse and work with the response data (assuming it's JSON)
            data = response.json()
            print(data)
        else:
            # Handle errors
            return {'status_code': response.status_code, 'error': response.text}
            

        text_data = "\n".join(f"Input Text: {entry['input_text']}\nTranslated Text: {entry['translated_text']}\n" for entry in data['results'])
        data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        file_name = "output.txt" 
        file_path = os.path.join(data_folder, file_name)

        # Store the text in a text file
        with open(file_path, "w") as file:
            file.write(text_data)

        loader = TextLoader(file_path)
        docs = loader.load()

        print('Splitting...')
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        print('Embedding...')
        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), client= self.chroma_client)
        
       