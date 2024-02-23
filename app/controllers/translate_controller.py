import traceback
from fastapi import HTTPException
from fastapi.responses import JSONResponse
from app.libs import GPTLanguageTranslator

class TranslateController:
    def __init__(self):
        pass

    def translate(self, target_language: None, text: None):
        try:
            translator = GPTLanguageTranslator()

            translated_text = translator.translate_text(text, target_language)

            return JSONResponse(content={'message': 'Text is translated successfully', 'target_language': target_language, 'text': text, 'translated_text': translated_text}, status_code=200)
        except Exception as e: 
            error_message = str(e)
            traceback_info = traceback.format_exc()
            raise HTTPException(status_code=500, detail={'error': error_message, 'traceback': traceback_info})