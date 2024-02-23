import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
load_dotenv()

class GPTLanguageTranslator:
    def __init__(self):
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        if not self.openai_api_key:
            raise ValueError("OpenAI API key is not set in the .env file.")
        
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

    def translate_text(self, text, target_language):
        template = """I want you to act as a language translator. 
        I will provide you a text and a target language. 
        You will translate the text into the target language. 
        You will return only the translated text.
        Text: {text}
        Target Language: {target_language}
        Translated Text: """

        prompt = ChatPromptTemplate.from_template(template)

        model = ChatOpenAI(model_name="gpt-4-0125-preview", temperature=0)

        chain = (
            { "target_language": RunnableLambda(lambda x: target_language), "text": RunnablePassthrough()}
            | prompt
            | model
            | StrOutputParser()
        )

        response = chain.invoke(text)

        return response

        
       