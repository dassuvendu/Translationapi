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
        
    async def train_model(self):
        try:
            translator = GPTLanguageTranslator()

            train_model = await translator.train_model()
            if 'error' in train_model and train_model['error']:
                return JSONResponse(content={'message': train_model['error']}, status_code=train_model['status_code'])
            return JSONResponse(content={'message': 'Model is trained successfully'}, status_code=200)
             
        except Exception as e: 
            error_message = str(e)
            traceback_info = traceback.format_exc()
            raise HTTPException(status_code=500, detail={'error': error_message, 'traceback': traceback_info})