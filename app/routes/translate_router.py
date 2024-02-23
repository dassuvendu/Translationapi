from typing import Annotated
from fastapi import APIRouter, Form
from app.controllers import TranslateController

router = APIRouter()
translate_controller = TranslateController()

@router.post("/translate/")
def translate(target_language: Annotated[str, Form()], text: Annotated[str, Form()]):
    return translate_controller.translate(target_language, text)
