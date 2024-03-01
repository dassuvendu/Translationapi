from fastapi import FastAPI
from app.routes import translate_router

app = FastAPI()

app.include_router(translate_router.router, prefix='/api')


