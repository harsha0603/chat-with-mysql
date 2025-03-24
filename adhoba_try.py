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
        "address": None, "room number": None, "name": None, "phone number": None, "problem": None
    }
if "last_room_results" not in st.session_state:
    st.session_state["last_room_results"] = []
if "current_step" not in st.session_state:
    st.session_state["current_step"] = "identify_user"
if "mrt_sub_step" not in st.session_state:
    st.session_state["mrt_sub_step"] = "choose_line"
if "mrt_stations" not in st.session_state:
    st.session_state["mrt_stations"] = [] 
if "current_mrt_index" not in st.session_state:
    st.session_state["current_mrt_index"] = 0  
if "last_llm_query" not in st.session_state:
    st.session_state["last_llm_query"] = ""  
if "last_modified_query" not in st.session_state:
    st.session_state["last_modified_query"] = ""  
if "max_budget" not in st.session_state:
    st.session_state["max_budget"] = 0  


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
    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"chat_history": chat_history, "user_query": user_query})
    return response.strip().lower()


def extract_existing_user_info(chat_history):
    template = """
You are Aba from Adobha Co-living. Extract the following from the conversation history for an existing user:
- address
- room number
- name
- phone number 
- problem

Conversation History:
{chat_history}

Extracted Info (return as JSON with null for missing fields):
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"chat_history": chat_history})
    cleaned_response = clean_response(response)
    try:
        return json.loads(cleaned_response)
    except json.JSONDecodeError:
        return {"address": None, "room number": None, "name": None, "phone number ": None, "problem": None}
    

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
  AND r.sellingprice > 0
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
    llm = ChatOpenAI(model="gpt-4o-mini")
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
You are Aba from Adobha Co-living. Based on the database schema, conversation history, and MRT station dictionary, generate a valid, single-line SQL query that finds rooms matching the user's preferences.  
**Return ONLY the SQL query itself—no additional text, comments, markdown, or explanations.**

## **Database Schema:**  
{schema}  

## **Conversation History:**  
{chat_history}  

## **User Question:**  
{question}  


## **Instructions:**  

### **1. Status Filtering**  
- Include only rooms where `rooms.occupancystatus` is `'Vacant'`, `'Available Soon'`, or `'Available Immediately'`.  
- Ensure `properties.status` is `'a'`.  

### **2. Required Columns**  
- **Building Name**: `MAX(properties.buildingname) AS BuildingName`  
- **Nearest MRT**: `MAX(properties.nearestmrt) AS NearestMRT`  
- **Rent**: `MAX(rooms.sellingprice) AS Rent`  
- **Amenities**: Include ALL `true` amenities from `rooms` using `MAX(CASE WHEN rooms.[amenity] = 'true' THEN 'AmenityName' END)` Always include `aircon`.  
- **Washroom Details**: `MAX(washrooms.size) AS WashroomSize`, `MAX(washrooms.location) AS WashroomLocation`  

### **3. User Preference Filters**  
Apply these when mentioned:  
- **Nearest MRT**: Use `properties.nearestmrt LIKE '%mrt%'` to allow partial matches. (Fuzzy Matching)
- **Budget**: Filter `rooms.sellingprice` with `BETWEEN min_budget AND max_budget`.  
- **Max Occupancy**:  
  - Ensure `rooms.maxoccupancy` accounts for variations like `'1 Pax'`, `'Single'`, `'Couple'`, etc.  
  - Use: `rooms.maxoccupancy = '1 Pax' OR rooms.maxoccupancy LIKE '%Single%'`.  (For single occupancy)
  - Use `rooms.maxoccupancy = '2 Pax' OR rooms.maxoccupancy LIJE '%Couple%' . (For 2 occupancy)
- **Washroom Preferences**: See below.  
- **Amenities**: Filter `rooms.[amenity] = 'true'` for each mentioned amenity.  

### **4. Washroom Preferences**  
- **"Private"**: `washrooms.location = 'Within Room'`.  
- **"Shared"**: `washrooms.location = 'Outside Room'`.   

### **5. Query Structure**  
- **Joins**: Use `JOIN properties ON rooms.propertyid = properties.propertyid` and `LEFT JOIN washrooms ON properties.propertyid = washrooms.propertyid`.  
- **Grouping**: Always `GROUP BY rooms.roomid`.  
- **Aggregation**: Apply `MAX()` to ALL non-grouped columns.  
- **LEFT JOIN Filters**: Place conditions in the `ON` clause with `COALESCE()`.  

### **6. Result Limit**  
- Default to `LIMIT 5` unless specified otherwise.  
---

**SQL Query:**  
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini")
    return (
        RunnablePassthrough.assign(
            schema=lambda _: db.get_table_info()
        )
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
    llm = ChatOpenAI(model="gpt-4o-mini")
    return prompt | llm | StrOutputParser()


def collect_existing_user_info_chain():
    template = """
You are Aba from Adobha Co-living. Collect the following from an existing user:
- Address
- Room number
- Name of the user
- Number of the user
- Problem 

Current info:
{info}

Conversation History:
{chat_history}

User’s latest query:
{question}

If the user provides info, thank them and ask for the next missing piece of information.
If all info is collected, say: "Thank you! I’ve got everything I need!."
If the user asks a question or goes off-topic, address it briefly and then ask for the next missing piece.

Aba’s response:
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini")
    return (
        RunnablePassthrough.assign(info=lambda _: st.session_state.get("existing_user_info", {}))
        | prompt
        | llm
        | StrOutputParser()
    )

def collect_new_user_info_chain():
    template = """
You are Aba from Adobha Co-living. Collect the following details from a new user one question at a time in a structured manner:

### **User Information Collection**

Ask the following questions one by one:

1. "What is your rental duration? (Minimum 3 months)"
   - If the response is less than 3 months, say: 
     "We kindly request a minimum rental duration of three months. Please ensure that the duration provided meets or exceeds this requirement."
   
2. "What is your pass type? (EP, Student, Work)"
    - If response is "Tourist", say:
      "Currently, we regret to inform you that our services are not available for tourists. We sincerely apologize for any inconvenience."

3. "What is your move-in date?"

4. "Do you prefer a private or shared washroom?"

5. "How many occupants will be staying?"

6. "What is your budget range? (Please provide min and max budget)"

7. "Which MRT Line are you interested in?"
   - Provide options: East-West Line (EW), Downtown Line (DT), North-South Line (NS), Circle Line (CC), North-East Line (NE), Thomson-East Coast Line (TE).
   - Store the response as preferred_mrt_line.

8. "Which station do you prefer?"
   - Based on the selected MRT line, show relevant stations:
     - EW: Kembangan, Simei, Chinese Garden, Eunos, Pasir Ris
     - DT: Upper Changi, Cashew, Hume, Mountbatten
     - NS: Admiralty, Novena, Orchard, Yio Chu Kang, Yew Tee
     - CC: Nicoll Highway, Holland Village, Mountbatten
     - NE: Woodleigh
     - TE: Lentor, Upper Changi East
   - Store the response as preferred_mrt.

9. "What is your nationality?"

10. "What is your gender?"

### **Guidelines for Interaction**
- If the user provides partial info, thank them and move to the next missing question.
- If all details are collected, say:  
  "Thank you! I’ve got everything I need. Let me find some great options for you!"
- If the user asks unrelated questions, respond briefly before steering back to the next missing question.

Current Preferences:
{preferences}

Conversation History:
{chat_history}

User’s Latest Query:
{question}

Aba’s Response:
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini")
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

import re
import streamlit as st

def modify_query_for_new_mrt(original_query, new_mrt_station):
    """Dynamically replaces the MRT station in the LLM-generated query."""
    
    pattern = r"(p\.nearestmrt\s*=\s*')([^']+)(')"
    modified_query = re.sub(pattern, fr"\1{new_mrt_station}\3", original_query)

    logger.info(f"Modified query for new MRT station: {new_mrt_station}")
    return modified_query

def modify_query_for_budget(original_query, new_max_budget):
    """Dynamically increases the max budget in the query."""
    
    pattern = r"(r\.sellingprice\s*<=\s*)(\d+)"
    modified_query = re.sub(pattern, fr"\1{new_max_budget}", original_query)

    logger.info(f"Relaxing budget: New Max Budget = {new_max_budget}")
    return modified_query


def fallback_logic():
    """Tries alternative MRT stations first, then relaxes budget if needed."""
    
    mrt_stations = st.session_state.get("mrt_stations", [])
    current_mrt_index = st.session_state.get("current_mrt_index", 0)
    last_query = st.session_state.get("last_modified_query", st.session_state.get("last_llm_query"))  
    max_budget = st.session_state.get("max_budget") 
    budget_step = 500  

    if current_mrt_index + 1 < len(mrt_stations):
        next_station = mrt_stations[current_mrt_index + 1]
        st.session_state["current_mrt_index"] += 1

        logger.info(f"Fallback: Trying next MRT station: {next_station}")

        modified_query = modify_query_for_new_mrt(last_query, next_station)
        st.session_state["last_modified_query"] = modified_query  # Store modified query

        results = execute_query(modified_query)
        if results["status"] == "success" and results["data"]:
            return {"status": "success", "data": results["data"]}
    if max_budget:
        new_budget = max_budget + budget_step  # Increase budget
        st.session_state["max_budget"] = new_budget  # Update session

        logger.info(f"Fallback: Relaxing budget to {new_budget}")

        modified_query = modify_query_for_budget(last_query, new_budget)
        st.session_state["last_modified_query"] = modified_query  # Store modified query

        results = execute_query(modified_query)
        if results["status"] == "success" and results["data"]:
            return {"status": "success", "data": results["data"]}

    return {"status": "fail", "message": "No alternative properties found."}


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
    llm = ChatOpenAI(model="gpt-4o-mini")
    return prompt | llm | StrOutputParser()

def get_response(user_query: str, chat_history: list):
    new_history = chat_history + [HumanMessage(content=user_query)]

    chain = classify_intent(st.session_state["chat_history"], user_query)
    intent = chain.invoke({
    "chat_history": st.session_state["chat_history"],
    "user_query": user_query
    }).strip().lower()

    logger.info(f"Classified intent: {intent}")
    
    # Check if we're in the process of identifying user type
    if st.session_state["current_step"] == "identify_user":
        # Determine if user is new or existing
        user_type = classify_user_type(st.session_state["chat_history"], user_query).strip().lower()
        logger.info(f"User type classified as: {user_type}")
        
        if user_type == "existing":
            st.session_state["user_type"] = "existing"
            st.session_state["current_step"] = "collect_existing_user_info"
            response = collect_existing_user_info_chain().invoke({
                "chat_history": st.session_state["chat_history"],
                "question": user_query
            })
        elif user_type == "new":
            st.session_state["user_type"] = "new"
            st.session_state["current_step"] = "collect_new_user_info"
            # Welcome message for new users
            response = "Welcome to Adobha Co-living! I'd be happy to help you find the perfect place to stay. Let me ask you a few questions to understand your preferences better.\n\nWhat is your rental duration? (Minimum 3 months)"
        else:
            # If unclear, ask again
            response = "I'm not sure if you're an existing user or a new user. Could you please clarify?"
    
    # Handle existing user information collection
    elif st.session_state["current_step"] == "collect_existing_user_info":
        # Extract existing user info
        extracted_info = extract_existing_user_info(st.session_state["chat_history"])
        st.session_state["existing_user_info"].update(
            {k: v for k, v in extracted_info.items() if v is not None}
        )
        
        # Check if we have all required info
        required_fields = ["address", "room number", "name", "problem"]
        if all(st.session_state["existing_user_info"].get(field) for field in required_fields):
            st.session_state["current_step"] = "handle_existing_user"
            
            
            response = "Thank you for providing your information. I've recorded your concern and will help address it. Is there anything specific you'd like to know about our policies or services in the meantime?"
        else:
            # Continue collecting info
            response = collect_existing_user_info_chain().invoke({
                "chat_history": st.session_state["chat_history"],
                "question": user_query
            })
    
    # Handle new user information collection
    elif st.session_state["current_step"] == "collect_new_user_info":
        # Extract preferences from conversation
        extracted_preferences = extract_preferences(new_history)
        st.session_state["preferences"].update(
            {k: v for k, v in extracted_preferences.items() if v is not None}
        )
        
        # Handle MRT line selection
        if st.session_state["mrt_sub_step"] == "choose_line" and st.session_state["preferences"].get("preferred_mrt_line"):
            st.session_state["mrt_sub_step"] = "choose_station"
            mrt_stations = get_mrt_stations(st.session_state["preferences"]["preferred_mrt_line"])
            response = f"Great! For the {st.session_state['preferences']['preferred_mrt_line']}, which station are you interested in? Options include: {', '.join(mrt_stations)}"
            
        # Check if we have all essential preferences
        essential_fields = ["rental_duration", "pass_type", "washroom_preference", "occupants", "min_budget", "max_budget", "preferred_mrt"]
        if all(st.session_state["preferences"].get(field) for field in essential_fields):
            st.session_state["current_step"] = "search_properties"
            
            # Generate SQL query based on preferences
            sql_query = generate_query_from_preferences(st.session_state["chat_history"], user_query)
            logger.info(f"Generated SQL query: {sql_query}")
            st.session_state["last_llm_query"] = sql_query
            
            # Execute query and get results
            results = execute_query(sql_query)
            st.session_state["last_room_results"] = results
            
            # Present results to user
            if results["status"] == "success":
                response = get_property_info_chain().invoke({
                    "results": results["data"],
                    "chat_history": st.session_state["chat_history"],
                    "question": user_query
                })
            else:
                response = results["message"]
        else:
            # Continue collecting preferences
            response = collect_new_user_info_chain().invoke({
                "chat_history": st.session_state["chat_history"],
                "question": user_query
            })
    
    # For users who have completed the initial information collection
    elif st.session_state["current_step"] in ["handle_existing_user", "search_properties"]:
        if intent == "property_search" and st.session_state["user_type"] == "new":
            # Extract any new preferences mentioned
            extracted_preferences = extract_preferences(st.session_state["chat_history"])
            st.session_state["preferences"].update(
                {k: v for k, v in extracted_preferences.items() if v is not None}
            )
            
            # Generate SQL query based on updated preferences
            sql_query = generate_query_from_preferences(st.session_state["chat_history"], user_query)
            logger.info(f"Generated SQL query: {sql_query}")
            
            # Execute query and get results
            results = execute_query(sql_query)
            st.session_state["last_room_results"] = results
            
            if results["status"] == "success" and results["data"]:
                response = get_property_info_chain().invoke({
                "results": results["data"],
                "chat_history": st.session_state["chat_history"],
                "question": user_query
            })
            else:
                fallback_results = fallback_logic(st.session_state["preferences"])
                if fallback_results["status"] == "success" and fallback_results["data"]:
                    response = get_property_info_chain().invoke({
                    "results": fallback_results["data"],
                    "chat_history": st.session_state["chat_history"],
                    "question": user_query
                })
                else:
                    response = fallback_results["message"]
        
        elif intent == "information_request":
            response = handle_faq_questions(user_query)
        
        elif intent == "viewing_request":
            # Check if we already have a viewing in progress
            if "viewing_details" not in st.session_state:
                st.session_state["viewing_details"] = {
                    "name": None,
                    "phone": None,
                    "date_time": None,
                    "property_address": None
                }
                st.session_state["current_step"] = "collect_viewing_info"
                response = "I'd be happy to schedule a viewing for you. First, could you please tell me your name?"
            else:
                response = handle_viewing_request(user_query)
        
        else:
            # General conversation
            response = generate_llm_response(
                st.session_state["chat_history"], 
                user_query,
                "You are Aba from Adobha Co-living. Respond to the user's message: {user_query}"
            )
    
    # Handle viewing request information collection
    elif st.session_state["current_step"] == "collect_viewing_info":
        response = handle_viewing_request(user_query)
    
    else:
        # Fallback for any other state
        response = generate_llm_response(
            st.session_state["chat_history"], 
            user_query,
            "You are Aba from Adobha Co-living. Respond to the user's message: {user_query}"
        )
    
    # Add assistant's response to chat history
    
    return response


def handle_faq_questions(user_query):
    """
    Handle FAQ questions about policies and services.
    
    Args:
        user_query: The user's question
        
    Returns:
        A response addressing the FAQ
    """
    # Define common FAQs and their answers
    faqs = {
        "cooking": "Yes, cooking is allowed in all our properties. Each unit is equipped with basic cooking facilities.",
        "visitor": "Yes, you can bring visitors. However, please inform us or seek permission before hosting anyone overnight.",
        "parking": "Yes, parking is available at most of our properties. Could you please share your vehicle plate number so we can arrange parking access?",
        "charges": "Besides rent, there are a few additional charges:\n- Housekeeping: S$30 monthly\n- Air conditioning service: S$60 quarterly\n- Utilities are typically charged based on usage"
    }
    
    # Use LLM to determine which FAQ the query relates to
    template = """
    Determine which category this question falls into:
    - cooking (if about cooking, food preparation)
    - visitor (if about guests, visitors)
    - parking (if about parking, vehicles)
    - charges (if about additional fees, costs beyond rent)
    - other (if doesn't match any category)
    
    User question: {user_query}
    
    Category (just the category word):
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4o-mini")
    chain = prompt | llm | StrOutputParser()
    
    category = chain.invoke({"user_query": user_query}).strip().lower()
    
    if category in faqs:
        return faqs[category]
    else:
        # For questions not covered in the FAQs
        return generate_llm_response(
            st.session_state["chat_history"],
            user_query,
            "You are Aba from Adobha Co-living. Answer the user's question about our policies or services. Be helpful and informative: {user_query}"
        )


def handle_viewing_request(user_query):
    viewing_details = st.session_state.get("viewing_details", {})
    
    # Collect name if not present
    if not viewing_details.get("name"):
        # Extract name from user query
        name_template = "Extract only the person's name from this text: {user_query}"
        name = generate_llm_response([], user_query, name_template)
        
        if name and len(name.split()) >= 1:
            viewing_details["name"] = name
            st.session_state["viewing_details"] = viewing_details
            return "Thank you! Now, could you please provide your phone number? We need a valid Singapore number to contact you."
        else:
            return "I didn't catch your name. Could you please tell me your name for the viewing appointment?"
    
    # Collect phone number if not present
    elif not viewing_details.get("phone"):
        # Extract and validate Singapore phone number
        phone_match = re.search(r'(\+65)?\s*[689]\d{7}', user_query)
        
        if phone_match:
            phone = phone_match.group(0)
            # Standardize format
            phone = re.sub(r'[^\d]', '', phone)
            if not phone.startswith('65') and len(phone) == 8:
                phone = '65' + phone
            
            viewing_details["phone"] = phone
            st.session_state["viewing_details"] = viewing_details
            
            # If we have property results, ask which one they want to view
            if st.session_state.get("last_room_results", {}).get("status") == "success":
                properties = st.session_state["last_room_results"]["data"]
                property_list = "\n".join([f"{i+1}. {prop.get('BuildingName', 'Property')} near {prop.get('NearestMRT', 'MRT')}" 
                                          for i, prop in enumerate(properties[:5])])
                return f"Great! Which property would you like to view?\n\n{property_list}\n\nPlease select by number or name."
            else:
                return "Thank you! Which property location are you interested in viewing?"
        else:
            return "I need a valid Singapore phone number starting with +65 Could you please provide a valid phone number?"
    
    # Collect property address if not present
    elif not viewing_details.get("property_address"):
        # Try to extract which property they want to view
        property_index = None
        
        # Check if they selected by number
        number_match = re.search(r'\b[1-5]\b', user_query)
        if number_match and st.session_state.get("last_room_results", {}).get("status") == "success":
            property_index = int(number_match.group(0)) - 1
            properties = st.session_state["last_room_results"]["data"]
            if 0 <= property_index < len(properties):
                viewing_details["property_address"] = properties[property_index].get("BuildingName", "Property")
            
        # If no number found or invalid, extract property name using LLM
        if not viewing_details.get("property_address"):
            property_template = """
            Based on the user's response, which property are they interested in viewing?
            If there's no clear property mentioned, respond with "unclear".
            User response: {user_query}
            """
            property_name = generate_llm_response([], user_query, property_template)
            
            if property_name.lower() != "unclear":
                viewing_details["property_address"] = property_name
        
        if viewing_details.get("property_address"):
            st.session_state["viewing_details"] = viewing_details
            return "When would you like to schedule the viewing? Please provide a date and time that works for you."
        else:
            return "I'm not sure which property you'd like to view. Could you please specify the building name or provide more details about the location?"
    
    # Collect date and time if not present
    elif not viewing_details.get("date_time"):
        # Extract date and time from user query
        datetime_template = """
        Extract the date and time for a viewing appointment from this text. 
        Format as: Day, DD Month at HH:MM AM/PM
        If unclear, respond with "unclear".
        Text: {user_query}
        """
        date_time = generate_llm_response([], user_query, datetime_template)
        
        if date_time.lower() != "unclear":
            viewing_details["date_time"] = date_time
            st.session_state["viewing_details"] = viewing_details
            
            # Confirmation message with all details
            confirmation = f"""
            Great! I've scheduled your viewing appointment with the following details:
            
            Name: {viewing_details['name']}
            Phone: {viewing_details['phone']}
            Property: {viewing_details['property_address']}
            Date & Time: {viewing_details['date_time']}
            
            Our property manager will contact you to confirm this appointment. Is there anything else you'd like to know about the property or the viewing process?
            """
            
            # Reset the viewing request process for future requests
            st.session_state["current_step"] = "search_properties"
            return confirmation
        else:
            return "I couldn't understand the date and time. Could you please provide it in a format like 'Monday, 25 March at 3:00 PM'?"
    
    # Fallback
    else:
        return "I have all the information I need for your viewing request. Is there anything else you'd like to know?"
    


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