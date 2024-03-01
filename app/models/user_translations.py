from sqlalchemy import create_engine, Column, Integer, Text, TIMESTAMP
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class UserTranslation(Base):
    __tablename__ = "user_translations"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    input_text = Column(Text(collation='utf8mb4_unicode_ci'))
    translated_text = Column(Text(collation='utf8mb4_unicode_ci'))
    created_at = Column(TIMESTAMP)
    updated_at = Column(TIMESTAMP)
