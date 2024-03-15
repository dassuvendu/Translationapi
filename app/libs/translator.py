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

        self.collection_name = "translations"

    def translate_text(self, text, target_language):
        chunks = []
        max_chunk_size = 4000
        chunks = []
        current_chunk = ""
        for word in text.split():
            if len(current_chunk) + len(word) <= max_chunk_size:
                current_chunk += (word + " ")
            else:
                chunks.append(current_chunk.strip())
                current_chunk = word + " "
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        final_text = ''
        for chunk in chunks:
            text = chunk
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

            model = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0, max_tokens=4096)

            def format_docs(docs):
                return "\n\n".join(doc.page_content for doc in docs)
            
            chain = (
                { "target_language": RunnableLambda(lambda x: target_language), "text": RunnablePassthrough(), "context": retriever | format_docs}
                | prompt
                | model
                | StrOutputParser()
            )

            response = chain.invoke(text)
            final_text += response

        return final_text

    async def train_model(self):
        url = "https://aitranslationhub.co/api/translations" 
        response = requests.get(url)
        if response.status_code == 200:
            data = response.json()
        else:
            return {'status_code': response.status_code, 'error': response.text}
            

        text_data = "\n".join(f"Text: {entry['input_text']}\nTranslated Text: {entry['translated_text']}\n" for entry in data['results'])
        data_folder = os.path.join(os.path.dirname(__file__), '..', 'data')
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)

        file_name = "output.txt" 
        file_path = os.path.join(data_folder, file_name)

        with open(file_path, "w", encoding="utf-8") as file:
            file.write(text_data)

        loader = TextLoader(file_path, encoding='utf-8')
        docs = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)

        coll = self.chroma_client.get_or_create_collection(self.collection_name)
        result = coll.get()
        if result:
            self.chroma_client.delete_collection(self.collection_name)

        vectorstore = Chroma.from_documents(documents=splits, embedding=OpenAIEmbeddings(), client=self.chroma_client, collection_name=self.collection_name)

        return {'message': 'Model is trained successfully'}
        
       