import os
import streamlit as st
import pandas as pd
import json
import re
from pathlib import Path
from dotenv import load_dotenv
import sqlparse
import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy import text


st.set_page_config(page_title="Aba Co-living Assistant", page_icon="🏠")
# Load env vars
load_dotenv()

# Path to the JSON file for preferences
PREFERENCES_FILE = Path("user_preferences.json")

def save_preferences_to_file(preferences: dict):
    """Save preferences to a JSON file."""
    with open(PREFERENCES_FILE, "w") as f:
        json.dump(preferences, f, indent=4)

def load_preferences_from_file():
    """Load preferences from a JSON file."""
    if PREFERENCES_FILE.exists():
        with open(PREFERENCES_FILE, "r") as f:
            return json.load(f)
    return {}

# Set up session state
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [AIMessage(content="Hi there! I’m Aba from Adobha Co-living. Are you an existing user or a new user?")]
if "preferences" not in st.session_state:
    st.session_state["preferences"] = {
        "rental_duration": None,
        "pass_type": None,
        "preferred_mrt": None,
        "washroom_preference": None,
        "occupants": None,
        "move_in_date": None,
        "nationality": None,
        "gender": None,
        "min_budget": None,
        "max_budget": None,
        "amenities": []
    }
# Session state initialization (updated with current_step)
if "chat_history" not in st.session_state:
    st.session_state["chat_history"] = [AIMessage(content="Hi there! I’m Aba from Adobha Co-living. Are you an existing user or a new user?")]
if "preferences" not in st.session_state:
    st.session_state["preferences"] = {
        "rental_duration": None, "pass_type": None, "move_in_date": None,
        "washroom_preference": None, "occupants": None, "min_budget": None, "max_budget": None,
        "preferred_mrt_line": None, "preferred_mrt": None,
        "nationality": None, "gender": None, "amenities": []
    }
if "collecting_info" not in st.session_state:
    st.session_state["collecting_info"] = True
if "pass_type" not in st.session_state:
    st.session_state["pass_type"] = None
if "work_study_location" not in st.session_state:
    st.session_state["work_study_location"] = None
if "nationality" not in st.session_state:
    st.session_state["nationality"] = None
if "gender" not in st.session_state:
    st.session_state["gender"] = None
if "user_type" not in st.session_state:
    st.session_state["user_type"] = None
if "existing_user_info" not in st.session_state:
    st.session_state["existing_user_info"] = {
        "address": None, "room_number": None, "problem": None
    }
if "last_room_results" not in st.session_state:
    st.session_state["last_room_results"] = []
if "current_step" not in st.session_state:
    st.session_state["current_step"] = "identify_user"
if "mrt_sub_step" not in st.session_state:
    st.session_state["mrt_sub_step"] = "choose_line"

# Database setup
@st.cache_resource
def init_database_engine():
    db_user = os.getenv("DB_USER")
    db_password = os.getenv("DB_PASSWORD")
    db_host = os.getenv("DB_HOST")
    db_port = os.getenv("DB_PORT", "3306")
    db_name = os.getenv("DB_NAME")

    if not all([db_user, db_password, db_host, db_port, db_name]):
        st.error("Missing some database credentials in the .env file!")
        raise ValueError("Database credentials not set!")

    db_uri = f"mysql+mysqlconnector://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
    return create_engine(db_uri)

@st.cache_resource
def init_database_session(_engine): # Add underscore to engine parameter
    return sessionmaker(bind=_engine)

@st.cache_resource
def init_sqldatabase(_engine): # Add underscore to engine parameter.
    return SQLDatabase(_engine)

engine = init_database_engine()
Session = init_database_session(engine)
db = init_sqldatabase(engine)

# --- Response Cleaning and Fallback Helpers ---
def clean_response(response: str) -> str:
    """Remove markdown code fences and any language identifier if present."""
    response = response.strip()
    response = re.sub(r"^```(?:json)?", "", response)
    response = re.sub(r"```$", "", response)
    return response.strip()

def fallback_mrt(chat_history: list, extracted: dict) -> dict:
    """If 'preferred_mrt' is null, scan the chat history for MRT mentions and update it."""
    if extracted.get("preferred_mrt") is None:
        for msg in reversed(chat_history):
            if hasattr(msg, "content"):
                text = msg.content.lower()
                if "east west" in text:
                    extracted["preferred_mrt"] = "East West Line"
                    break
                elif "simei" in text:
                    extracted["preferred_mrt"] = "Simei MRT"
                    break
    return extracted


def classify_user_type(chat_history, user_query):
    template = """
You are Aba from Adobha Co-living. Classify the user's response as "existing" or "new".

Conversation History:
{chat_history}

User's Message:
{user_query}

Response (existing, new, or unclear):
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-0125-preview")
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"chat_history": chat_history, "user_query": user_query})
    return response.strip().lower()


def extract_existing_user_info(chat_history):
    template = """
You are Aba from Adobha Co-living. Extract the following from the conversation history for an existing user:
- address
- room_number
- problem

Conversation History:
{chat_history}

Extracted Info (return as JSON with null for missing fields):
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-0125-preview")
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"chat_history": chat_history})
    cleaned_response = clean_response(response)
    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        return {"address": None, "room_number": None, "problem": None}
    

from datetime import datetime
import dateutil.parser

def standardize_date(date_str):
    """Standardize date string to YYYY-MM-DD, assuming future dates if year is omitted."""
    try:
        parsed_date = dateutil.parser.parse(date_str, fuzzy=True)
        current_year = datetime.now().year
        current_date = datetime.now()
        if parsed_date.year == current_year and parsed_date < current_date:
            parsed_date = parsed_date.replace(year=current_year + 1)
        return parsed_date.strftime("%Y-%m-%d")
    except ValueError:
        return None

def manual_parse_preferences(user_input):
    preferences = {}
    parts = [part.strip() for part in user_input.split(",")]
    # Step 1
    if len(parts) >= 3:
        if re.match(r"^\d+\s*(months?|years?)", parts[0], re.IGNORECASE):
            preferences["rental_duration"] = parts[0]
        if parts[1].upper() in ["EP", "STUDENT", "WORK"]:
            preferences["pass_type"] = parts[1].upper()
        standardized_date = standardize_date(parts[2])
        if standardized_date:
            preferences["move_in_date"] = standardized_date
    # Step 2
    if len(parts) >= 6:
        if parts[3].lower() in ["private", "shared"]:
            preferences["washroom_preference"] = parts[3].lower()
        if parts[4].isdigit():
            preferences["occupants"] = parts[4]
        budget_part = parts[5].replace("to", "-").replace(" ", "")
        budget_match = re.match(r"(\d+)-(\d+)", budget_part)
        if budget_match:
            preferences["min_budget"], preferences["max_budget"] = budget_match.groups()
    return preferences

def get_mrt_stations(mrt_line):
    if mrt_line is None:
        logger.warning("mrt_line is None in get_mrt_stations—user hasn’t specified an MRT line yet.")
        return []

    mrt_line_mappings = {
        "east west line": "EW",
        "east-west line": "EW",
        "downtown line": "DT",
        "north south line": "NS",
        "circle line": "CC",
        "northeast line": "NE",
        "thomson-east coast line": "TE"
    }

    mrt_line_key = mrt_line.lower() if isinstance(mrt_line, str) else mrt_line
    mrt_line_code = mrt_line_mappings.get(mrt_line_key, mrt_line_key) if mrt_line_key else "%"

    # Handle both bracketed and non-bracketed formats
    pattern = f"%({mrt_line_code})%"  # Bracketed format
    pattern_alt = f"% {mrt_line_code}%"  # Non-bracketed format

    query = """
    SELECT DISTINCT p.nearestmrt 
    FROM properties p 
    JOIN rooms r ON r.propertyid = p.propertyid 
    WHERE (p.nearestmrt LIKE :pattern OR p.nearestmrt LIKE :pattern_alt)
      AND r.occupancystatus IN ('Vacant', 'Available Soon', 'Available Immediately') 
      AND p.status = 'a'
    ORDER BY p.nearestmrt
    """

    try:
        with engine.connect() as conn:
            result = conn.execute(text(query), {"pattern": pattern, "pattern_alt": pattern_alt}).fetchall()
            return [row[0] for row in result] if result else []
    except Exception as e:
        logger.error(f"Error fetching MRT stations: {e}")
        return []


def extract_preferences(chat_history):
    chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
    template = f"""
    You are Aba from Adobha Co-living. Extract preferences from the conversation history:

    - rental_duration (e.g., "4 months")
    - pass_type (EP, Student, Work)
    - move_in_date (YYYY-MM-DD, assume future dates if year omitted)
    - washroom_preference (private, shared)
    - occupants (number, e.g., "1")
    - min_budget (numeric, e.g., "1000")
    - max_budget (numeric, e.g., "2000")
     - preferred_mrt_line (e.g., "EW", "DT"; map "East West Line" or "East-West Line" to "EW", "Downtown Line" to "DT")
    - preferred_mrt (e.g., "Simei EW")
    - nationality (e.g., "Singaporean")
    - gender (e.g., "male")

    Instructions:
    - Standardize move_in_date to YYYY-MM-DD.
    - Split budget range (e.g., "1000-2000") into min_budget and max_budget.
    - Return JSON with null for missing fields.

    Conversation History:
    {chat_history_str}

    Extracted Preferences:
    """
    llm = ChatOpenAI(model="gpt-4-0125-preview")
    response = llm.invoke(template)
    cleaned_response = clean_response(response.content)
    try:
        preferences = json.loads(cleaned_response)
    except json.JSONDecodeError:
        preferences = {k: None for k in st.session_state["preferences"].keys()}
    
    # Manual fallback
    user_input = chat_history[-1].content if chat_history else ""
    manual_extracted = manual_parse_preferences(user_input)
    preferences.update({k: v for k, v in manual_extracted.items() if v is not None})
    return preferences

# --- Dynamic SQL Query Generation using LLM ---
def get_sql_chain(db):
    template = """
You are Aba from Adobha Co-living. Based on the database schema provided and the conversation history, generate a valid, single-line SQL query that finds rooms matching the user's preferences.  
**Crucially, return ONLY the SQL query itself. Do NOT include any additional text, comments, markdown code blocks, or explanations.**  
Do not add any additional text, only the SQL query.  

## **Database Schema:**  
{schema}  

## **Conversation History:**  
{chat_history}  

## **Question:**  
{question}  

---

## **Important Instructions:**  

### **1. Status Handling**  
Only include rooms where `rooms.occupancystatus` is either `'Vacant'`, `'Available Soon'`, or `'Available Immediately'`, and the associated property's status (`properties.status`) is `'a'`.  

### **2. Room Details**  
Include only:  
- **Building Name** → `MAX(properties.buildingname) AS BuildingName`  
- **Nearest MRT** → `MAX(properties.nearestmrt) AS NearestMRT`
- **Rent** → `MAX(rooms.sellingprice) AS Rent`  
- **Amenities** → ALL `true` amenities from `rooms`(Dont forget to include aircon)  
- **Washroom details** → `MAX(washrooms.size) AS WashroomSize`, `MAX(washrooms.location) AS WashroomLocation`  

### **3. Filter Requirements**  
ALWAYS include these filters when mentioned by the user:  
- **Nearest MRT station** → Use `properties.nearestmrt LIKE '%station_name%'`  
- **Budget Range** → Map to `rooms.sellingprice` (NOT `rentmonth`), using `BETWEEN min_budget AND max_budget`  
- **Max Occupancy** → Use `rooms.maxoccupancy`  
- **Washroom Preferences** (see below)  
- **Any amenities explicitly mentioned by the user**  

### **4. Washroom Preferences Filtering (ALWAYS INCLUDE when mentioned)**  
- If **"private"**, filter rooms where `washroomno = 1` AND `washrooms.location = 'Within Room'`.  
- If **"shared"**, filter rooms where `washroomno > 1` OR `washrooms.location != 'Within Room'`.  
- If `totalusers` is mentioned, use `COALESCE(washrooms.totalusers, 0) > X` in the `JOIN` condition, NOT in `WHERE`.  

### **5. 🚨 CRITICAL: Aggregation for Non-Grouped Columns**  
- When using `GROUP BY rooms.roomid`, you MUST apply **aggregate functions (`MAX()`, `MIN()`)** to ALL non-grouped columns.  
- Use `MAX()` for all property columns, room columns, and washroom columns.  
- For amenities, use `MAX(CASE WHEN rooms.amenity = 'true' THEN 'AmenityName' END)` for EACH amenity.  
- Include ALL available amenities in the database, not just a subset.  

### **6. 🚨 LEFT JOIN Filtering**  
- **DO NOT** place `LEFT JOIN` filtering conditions in the `WHERE` clause.  
- Always move `LEFT JOIN` filters to the `ON` clause using `COALESCE()` for NULL handling.  

### **7. Limit Results**  
Default to **5 rows** unless the user specifies otherwise.  

---

## 🚨 **Key Fixes Applied:**  
✅ **Nearest MRT** is now `properties.nearestmrt`  
✅ **Budget filtering** is always based on `rooms.sellingprice`, NOT `rentmonth`.  

---
 **SQL Query:**  

""" 
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-0125-preview")
    return (
        RunnablePassthrough.assign(schema=lambda _: db.get_table_info())
        | prompt
        | llm
        | StrOutputParser()
    )



def clean_sql_query(query: str) -> str:
    """Remove markdown code fences and any language identifier from the SQL query."""
    query = query.strip()
    query = re.sub(r"^```(?:sql)?\s*", "", query)
    query = re.sub(r"\s*```$", "", query)
    query = query.replace("sql\n", "")
    query = query.replace("```sql", "") #Added to remove ```sql
    return query.strip()

def validate_sql_syntax(sql_query: str) -> bool:
    """Validate SQL syntax."""
    try:
        sqlparse.parse(sql_query)
        return True
    except Exception:
        return False


def generate_query_from_preferences(chat_history: list, user_question: str):
    # Use the dynamic SQL chain that takes schema, conversation history, and latest question
    sql_chain = get_sql_chain(db)

    chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

    preferences = st.session_state["preferences"]

    prompt = f"""
    You are a SQL query generator. Generate a SQL query to retrieve rooms based on user preferences.

    User Preferences:
    {preferences}

    Conversation History:
    {chat_history_str}

    User's Question:
    {user_question}

    Generate the SQL query:
    """

    response = sql_chain.invoke({
        "chat_history": chat_history_str,
        "question": prompt 
    })

    # Strip and clean the response
    raw_query = response.strip()
    sql_query = clean_sql_query(raw_query)
    return sql_query


import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.exc import SQLAlchemyError # Import SQLAlchemyError

# Assuming 'engine' is already defined elsewhere (e.g., create_engine('mysql+mysqlconnector://user:password@host/database'))

def execute_query(query: str):
    try:
        connection = engine.raw_connection()
        try:
            cursor = connection.cursor()
            try:
                print(f"Executing Query: {query}") #Debug print.
                cursor.execute(query)
                columns = [col[0] for col in cursor.description]
                rows = cursor.fetchall()
                df = pd.DataFrame(rows, columns=columns)
                if df.empty:
                    return {"status": "no_matches", "message": "I couldn't find any properties matching your preferences. Want to tweak them a bit?"}
                return {"status": "success", "data": df.to_dict(orient="records")}
            finally:
                cursor.close()
        finally:
            connection.close()
    except Exception as e:
        return {"status": "error", "message": f"Oops, something went wrong with the query: {e}"}

# --- Response Chains for Final Answer ---
def get_property_info_chain():
    template = """
You are Aba from Adobha Co-living. Present room details:

<RESULTS>
{results}
</RESULTS>

Conversation History:
{chat_history}

User’s latest query:
{question}

List rooms with:
- Building name
- Nearest MRT
- Rent
- Amenities (list as bullet points, e.g., - Aircon: Yes)
- Available from
Use bullet points for each room. Ask if they want to schedule a viewing.
If no matches, say: "I couldn’t find any rooms matching your preferences. Want to tweak them?"
Do not mention the <RESULTS> tags.

Aba’s response:
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-0125-preview")
    return prompt | llm | StrOutputParser()


def collect_existing_user_info_chain():
    template = """
You are Aba from Adobha Co-living. Collect the following from an existing user:
- Address
- Room number
- Problem

Current info:
{info}

Conversation History:
{chat_history}

User’s latest query:
{question}

If the user provides info, thank them and ask for the next missing piece of information.
If all info is collected, say: "Thank you! I’ve got everything I need. How can I assist with your problem?"
If the user asks a question or goes off-topic, address it briefly and then ask for the next missing piece.

Aba’s response:
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-0125-preview")
    return (
        RunnablePassthrough.assign(info=lambda _: st.session_state.get("existing_user_info", {}))
        | prompt
        | llm
        | StrOutputParser()
    )

def collect_new_user_info_chain():
    template = """
You are Aba from Adobha Co-living. Collect the following details from a new user in a structured manner:

### **Step 1: Basic Details**
- Rental duration (must be at least 3 months)
- Pass type (EP, Student, Work)
- Move-in date

### **Step 2: Room Preferences**
- Washroom preference (private or shared)
- Number of occupants
- Budget range (min and max)

### Step 3: MRT Selection
Ask the user:
"Which MRT Line are you interested in?"

Provide options: East-West Line (EW), Downtown Line (DT), North-South Line (NS), Circle Line (CC), North-East Line (NE), Thomson-East Coast Line (TE).
Store the response as preferred_mrt_line (e.g., "EW", "DT", "NS").
Based on the user’s choice, show the relevant MRT stations:

East-West Line (EW) → Kembangan (EW), Simei (EW), Chinese Garden (EW), Eunos (EW), Pasir Ris (EW).
Downtown Line (DT) → Upper Changi (DT), Cashew (DT), Hume (DT), Mountbatten (DT).
North-South Line (NS) → Admiralty (NS), Novena (NS), Orchard (NS), Yio Chu Kang (NS), Yew Tee (NS).
Circle Line (CC) → Nicoll Highway (CC), Holland Village (CC), Mountbatten (CC).
North-East Line (NE) → Woodleigh (NE).
Thomson-East Coast Line (TE) → Lentor (TE), Upper Changi East (TE).
Ask the user to choose a station from the selected MRT line.

Store the response as preferred_mrt (e.g., "Simei (EW)").

### **Step 4: Additional Details**
- Nationality 
- Gender

### **Guidelines for Interaction**
- If rental duration is **less than 3 months**, say:  
   "We require a minimum rental duration of 3 months. Please provide a duration of at least 3 months."
- If the user provides partial info, thank them and ask for the next missing detail.
- If all details are collected, say:  
   "Thank you! I’ve got everything I need. Let me find some great options for you!"
- If the user asks unrelated questions, respond briefly before steering back to the next missing detail.

Current Preferences:
{preferences}

Conversation History:
{chat_history}

User’s Latest Query:
{question}

Aba’s Response:
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-0125-preview")
    return (
        RunnablePassthrough.assign(preferences=lambda _: st.session_state.get("preferences", {}))
        | prompt
        | llm
        | StrOutputParser()
    )


def generate_llm_response(chat_history, user_query, prompt_template, **kwargs):
    prompt = ChatPromptTemplate.from_template(prompt_template)
    
    # Use gpt-4o-mini instead of gpt-4-0125-preview
    llm = ChatOpenAI(model="gpt-4o-mini")
    
    chain = prompt | llm | StrOutputParser()
    
    # Create a variables dictionary with required fields and any additional kwargs
    variables = {
        "chat_history": chat_history,
        "user_query": user_query,
        **kwargs  # This unpacks any additional variables needed by the template
    }
    
    # Invoke chain with all variables
    response = chain.invoke(variables)
    return response

def classify_intent(chat_history, user_query):
    template = """
    Classify the intent: 
    - general
    - information_request
    - property_search
    - viewing_request
    - error

    Conversation History:
    {chat_history}

    User’s Message:
    {user_query}

    Intent:
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-0125-preview")
    return prompt | llm | StrOutputParser()

def generate_dynamic_prompt(step, preferences, chat_history, user_query):
    llm = ChatOpenAI(model="gpt-4-0125-preview")
    chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

    if step == "step1_basic_details":
        missing = [f for f in ["rental_duration", "pass_type", "move_in_date"] if preferences[f] is None]
        template = f"""
        You are Aba from Adobha Co-living. Generate a friendly, natural question asking for:
        - How long they want to rent for
        - Pass type (EP, Student, Work)
        - Move-in date
        Current preferences: {preferences}
        Missing: {missing}
        History: {chat_history_str}
        User query: {user_query}
        Return only the question/response, no extra text.
        """
        return llm.invoke(template).content.strip()

    elif step == "step2_room_preferences":
        missing = [f for f in ["washroom_preference", "occupants", "min_budget", "max_budget"] if preferences[f] is None]
        template = f"""
        You are Aba from Adobha Co-living. Acknowledge Step 1 prefs and ask for:
        - Washroom preference (private or shared)
        - Number of occupants
        - Budget range (min and max)
        Current preferences: {preferences}
        Missing: {missing}
        History: {chat_history_str}
        User query: {user_query}
        Return only the question/response, no extra text.
        """
        return llm.invoke(template).content.strip()

    
    elif step == "step3_mrt_preferences":
        if "mrt_sub_step" not in st.session_state:
            return "Hmm, something went wrong. Let’s try that again—what are your preferences?"
    
    if st.session_state["mrt_sub_step"] == "choose_line":
        template = f"""
        You are Aba from Adobha Co-living. Acknowledge Step 2 preferences and ask:
        
        - Which MRT line do you prefer? Here are the available options:
          - East-West Line (EW)
          - Downtown Line (DT)
          - North-South Line (NS)
          - Circle Line (CC)
          - North-East Line (NE)
          - Thomson-East Coast Line (TE)
        
        Current preferences: {preferences}
        History: {chat_history_str}
        User query: {user_query}
        Return only the question/response, no extra text.
        """
    
    elif st.session_state["mrt_sub_step"] == "choose_station":
        mrt_line = preferences.get("preferred_mrt_line", "")
        stations = get_mrt_stations(mrt_line)
        
        if not stations:
            return "Looks like no stations are available on that line right now. Want to pick a different line?"
        
        stations_list = ", ".join(stations)
        template = f"""
        You are Aba from Adobha Co-living. Show available MRT stations and ask:
        - Available stations on {mrt_line}: {stations_list}
        - Which station they prefer
        
        Current preferences: {preferences}
        History: {chat_history_str}
        User query: {user_query}
        Return only the question/response, no extra text.
        """

    else:
        return "Hmm, something went wrong. Let’s try that again—what are your preferences?"

    return llm.invoke(template).content.strip()

def get_response(user_query: str, chat_history: list):
    new_history = chat_history + [HumanMessage(content=user_query)]
    
    # Step 1: User type
    if st.session_state["user_type"] is None:
        user_type = classify_user_type(chat_history, user_query)
        st.session_state["user_type"] = user_type
        if user_type == "existing":
            return "Hi! Let's sort out your issue—could you share your address?"
        elif user_type == "new":
            st.session_state["current_step"] = "step1_basic_details"
            return generate_dynamic_prompt("step1_basic_details", st.session_state["preferences"], new_history, user_query)
        return "Not sure if you're new or existing—could you clarify?"

    # Step 2: Extract preferences
    extracted = extract_preferences(new_history)
    st.session_state["preferences"].update(extracted)

    # Step 3: Intent
    intent_chain = classify_intent(new_history, user_query)
    intent = intent_chain.invoke({"chat_history": new_history, "user_query": user_query}).strip().lower()

    # Step 4: Handle intents
    if intent == "viewing_request":
        viewing_prompt = """
        You are Aba from Adobha Co-living. Handle a viewing request:
        - Ask for full name, phone number (+65 or 65 followed by 8 digits), date/time.
        - Provide room address, arrival instructions, and gratitude.
        History: {chat_history}
        Query: {user_query}
        """
        return generate_llm_response(new_history, user_query, viewing_prompt)
    
    elif intent == "information_request" and st.session_state.get("last_room_results"):
        info_prompt = """
        You are Aba from Adobha Co-living. You're having a natural conversation with a potential resident about housing.
        
        History: {chat_history}
        Query: {user_query}
        Preferences: {preferences}
        Room Results: {room_results}
        BuildingName: {BuildingName}
        rental_duration: {rental_duration}
        
        Guidelines:
        - Answer questions about previously mentioned properties directly and conversationally.
        - DO NOT generate SQL queries for follow-up questions about properties already mentioned.
        - Use the Room Results provided to answer specific questions about properties.
        - Use a friendly, helpful tone that feels like talking to a knowledgeable friend, not a form-filling bot.
        - If you don't know specific details about a property, offer to find out rather than making up information.
        - Common information you can provide includes: pet policies, cooking rules, visitor policies, parking, additional charges, and lease terms.
        - Let the conversation flow naturally - don't force the user down a predetermined path.
        - Only suggest next steps when it feels natural in the conversation, not after every response.
        
        Remember: The goal is to help the user find their ideal home through natural conversation, not to check boxes on a form.
        """
        
        # Extract building name from results if available
        building_name = None
        if st.session_state.get("last_room_results") and len(st.session_state["last_room_results"]) > 0:
            if "BuildingName" in st.session_state["last_room_results"][0]:
                building_name = st.session_state["last_room_results"][0]["BuildingName"]
        
        variables = {
    "chat_history": new_history,
    "user_query": user_query,
    "preferences": st.session_state["preferences"],
    "room_results": st.session_state["last_room_results"],
    "rental_duration": st.session_state["preferences"].get("rental_duration", "your stay")
}

# Only add BuildingName if it exists
        if building_name:
            variables["BuildingName"] = building_name
        
        # Pass all needed variables to the function
        return generate_llm_response(
            new_history, 
            user_query, 
            info_prompt,
            preferences=st.session_state["preferences"],
            room_results=st.session_state["last_room_results"],
            BuildingName=building_name,
            rental_duration=rental_duration
        )

    # Step 5: Handle user types
    if st.session_state["user_type"] == "existing":
        extracted_info = extract_existing_user_info(new_history)
        st.session_state["existing_user_info"].update(extracted_info)
        missing = [k for k, v in st.session_state["existing_user_info"].items() if v is None]
        if not missing:
            return "Thanks! I've got all I need—how can I help with your problem?"
        return f"Gotcha! What's your {missing[0]}?"

    elif st.session_state["user_type"] == "new":
        # Validate rental duration
        rental_duration = st.session_state["preferences"].get("rental_duration")
        if rental_duration and "month" in rental_duration.lower():
            try:
                months = int(rental_duration.split()[0])
                if months < 3:
                    st.session_state["preferences"]["rental_duration"] = None
                    return "We need at least 3 months—could you update your rental duration?"
            except ValueError:
                pass

        # Step-based flow with dynamic prompts
        if st.session_state["current_step"] == "step1_basic_details":
            missing = [f for f in ["rental_duration", "pass_type", "move_in_date"] if st.session_state["preferences"][f] is None]
            if missing:
                return generate_dynamic_prompt("step1_basic_details", st.session_state["preferences"], new_history, user_query)
            st.session_state["current_step"] = "step2_room_preferences"
            return generate_dynamic_prompt("step2_room_preferences", st.session_state["preferences"], new_history, user_query)

        elif st.session_state["current_step"] == "step2_room_preferences":
            missing = [f for f in ["washroom_preference", "occupants", "min_budget", "max_budget"] if st.session_state["preferences"][f] is None]
            if missing:
                return generate_dynamic_prompt("step2_room_preferences", st.session_state["preferences"], new_history, user_query)
            st.session_state["current_step"] = "step3_mrt_preferences"
            return generate_dynamic_prompt("step3_mrt_preferences", st.session_state["preferences"], new_history, user_query)

        elif st.session_state["current_step"] == "step3_mrt_preferences":
            if st.session_state["mrt_sub_step"] == "choose_line":
                if st.session_state["preferences"]["preferred_mrt_line"]:
                    st.session_state["mrt_sub_step"] = "choose_station"
                    return generate_dynamic_prompt("step3_mrt_preferences", st.session_state["preferences"], new_history, user_query)
                return generate_dynamic_prompt("step3_mrt_preferences", st.session_state["preferences"], new_history, user_query)
            
            elif st.session_state["mrt_sub_step"] == "choose_station":
                if st.session_state["preferences"]["preferred_mrt"]:
                    # Both MRT line and station are set—proceed to room search
                    sql_query = generate_query_from_preferences(new_history, user_query)
                    results = execute_query(sql_query)
                    if results["status"] == "success" and results["data"]:
                        st.session_state["last_room_results"] = results["data"]
                        response = get_property_info_chain().invoke({"results": results["data"], "chat_history": chat_history, "question": user_query})
                        return response + "\nThese rooms are available now—let me know if they work for your move-in plans!"
                    else:
                        relaxed_query = sql_query.replace(f"AND r.sellingprice BETWEEN {st.session_state['preferences']['min_budget']} AND {st.session_state['preferences']['max_budget']}", "")
                        results = execute_query(relaxed_query)
                        if results["status"] == "success" and results["data"]:
                            st.session_state["last_room_results"] = results["data"]
                            return f"No rooms match your budget, but here's what's close:\n" + get_property_info_chain().invoke({"results": results["data"], "chat_history": chat_history, "question": user_query})
                        return "I couldn't find any rooms near your chosen MRT station. Want to adjust your budget or pick another station?"
                return generate_dynamic_prompt("step3_mrt_preferences", st.session_state["preferences"], new_history, user_query)
    
    # Default fallback response
    return "I'm not sure how to help with that. Could you rephrase or provide more details?"

# Custom CSS with fixed sidebar visibility
custom_css = """
<style>
    /* Set the background color of the entire app to white */
    .stApp {
        background-color: #FFFFFF;  /* Pure White */
    }

    /* Set the text color to black for better readability in main window */
    .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5, .stApp h6, 
    .stApp p, .stApp div, .stApp span {
        color: #2D2D2D;  /* Dark Gray for better contrast */
    }

    /* Style the sidebar with a blue background */
    [data-testid="stSidebar"] {
        background-color: #4A90E2;  /* Professional Medium Blue */
    }

    /* Set the sidebar text color to white for high visibility */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3, 
    [data-testid="stSidebar"] h4, 
    [data-testid="stSidebar"] h5, 
    [data-testid="stSidebar"] h6, 
    [data-testid="stSidebar"] p, 
    [data-testid="stSidebar"] div, 
    [data-testid="stSidebar"] span {
        color: #FFFFFF !important;  /* White for maximum contrast, with !important to override defaults */
    }

    /* Style the reset button */
    .stButton button {
        background-color: #FF6B6B;  /* Vibrant Red for contrast */
        color: #FFFFFF;
        border: none;
        padding: 10px 20px;
        border-radius: 8px;
        transition: background-color 0.3s ease;
    }
    
    .stButton button:hover {
        background-color: #E55A5A;  /* Darker Red on hover */
    }

    /* Style the chat input box */
    .stChatInput input {
        background-color: #F8F9FA;  /* Very Light Gray */
        color: #2D2D2D;
        border: 1px solid #D1D9E6;
        border-radius: 8px;
        padding: 10px;
    }

    /* Style chat messages */
    .stChatMessage {
        border-radius: 10px;
        padding: 10px;
        margin: 5px 0;
    }

    /* AI message styling */
    [data-testid="stChatMessage"][data-testid="AI"] {
        background-color: #F0F4FF;  /* Very Light Blue for AI messages */
    }

    /* Human message styling */
    [data-testid="stChatMessage"][data-testid="Human"] {
        background-color: #F8F9FA;  /* Very Light Gray for human messages */
    }
</style>
"""

# Inject the custom CSS into the app
st.markdown(custom_css, unsafe_allow_html=True)

# --- UI Setup ---
st.title("GenZI Care Chat Bot")

# --- Sidebar ---
with st.sidebar:
    st.subheader("About Adobha Co-living")
    st.write("Singapore’s pioneer in co-living since 2013.")
    st.subheader("Our Offerings")
    st.write("• We provide a variety of fully furnished rooms with flexible lease terms and all-inclusive pricing.")
    st.subheader("Developed by GenZI care")
    if st.button("Reset Chat"):
        st.session_state["chat_history"] = [AIMessage(content="Hi there! I’m Aba from Adobha Co-living. Are you an existing user or a new user?")]
        st.session_state["preferences"] = {
            "rental_duration": None, "pass_type": None, "preferred_mrt": None,
            "washroom_preference": None, "occupants": None, "move_in_date": None,
            "nationality": None, "gender": None, "min_budget": None, "max_budget": None,
            "amenities": []
        }
        st.session_state["collecting_info"] = False
        st.session_state["user_type"] = None
        st.session_state["existing_user_info"] = {"address": None, "room_number": None, "problem": None}
        st.rerun()

# --- Chat UI ---
for message in st.session_state["chat_history"]:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar="👩‍💼"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human", avatar="👤"):
            st.markdown(message.content)

# --- Chat Input ---
user_query = st.chat_input("Drop your message here...")
if user_query and user_query.strip():
    st.session_state["chat_history"].append(HumanMessage(content=user_query))
    with st.chat_message("Human", avatar="👤"):
        st.markdown(user_query)
    with st.chat_message("AI", avatar="👩‍💼"):
        response = get_response(user_query, st.session_state["chat_history"])
        st.markdown(response)
    st.session_state["chat_history"].append(AIMessage(content=response))