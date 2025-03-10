import os
import streamlit as st
import pandas as pd
import json
from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv




def init_database() -> SQLDatabase:
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME")

    if not all([db_user, db_password, db_host, db_port, db_name]):
        st.error("Missing some database credentials in the .env file!")
        raise ValueError("Database credentials not set!")
    
    db_uri = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return SQLDatabase.from_uri(db_uri)

try:
    db = init_database()
    print("Connection successfull")
except Exception as e:
    st.error(f"Database connection failed: {e}")
    db = None


if __name__ == "__main__":
    init_database()