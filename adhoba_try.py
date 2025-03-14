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
        "address": None,
        "room_number": None,
        "problem": None
    }
if "last_room_results" not in st.session_state:
    st.session_state["last_room_results"] = []

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
    

# --- Extraction of Preferences ---
def extract_preferences(chat_history):
    chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])
    template = """
You are Aba from Adobha Co-living. Extract the following preferences for a new user:
- rental_duration (e.g., "6 months")
- pass_type (e.g., "EP", "Student", "Work")
- preferred_mrt (e.g., "Simei MRT"; derive from work location if mentioned)
- washroom_preference (e.g., "private", "shared")
- occupants (e.g., "1")
- move_in_date (e.g., "May 1st")
- nationality (e.g., "Singaporean")
- gender (e.g., "male", "female", "other")
- min_budget (e.g., "1000")
- max_budget (e.g., "2000")
- amenities (e.g., ["aircon", "wifi", "gym"])

Conversation History:
{chat_history}

Extracted Preferences (return as JSON with null for missing fields and empty list for amenities):
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-0125-preview")
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"chat_history": chat_history_str})
    cleaned_response = clean_response(response)
    try:
        preferences = json.loads(cleaned_response)
    except json.JSONDecodeError:
        preferences = {
            "rental_duration": None, "pass_type": None, "preferred_mrt": None,
            "washroom_preference": None, "occupants": None, "move_in_date": None,
            "nationality": None, "gender": None, "min_budget": None, "max_budget": None,
            "amenities": []
        }
    preferences = fallback_mrt(chat_history, preferences)  # Reuse existing fallback
    return preferences

# --- Dynamic SQL Query Generation using LLM ---
def get_sql_chain(db):
    template = """
You are Aba from Adobha Co-living. Based on the database schema provided and the conversation history, generate a valid, single-line SQL query that finds rooms matching the user's preferences.
**Crucially, return ONLY the SQL query itself. Do NOT include any additional text, comments, markdown code blocks, or explanations.**
Do not add any additional text, only the sql query.
Database Schema:
{schema}

Conversation History:
{chat_history}

Question:
{question}

**Important Instructions:**
1. **Date Handling**: Use DATE() for date comparisons. Filter out rooms where eavaildate < move-in date.
2. **Status Handling**: Only include rooms where rooms.status = 'a' and properties.status = 'a'.
3. **Room Details**: Include only:
  - Building name (buildingname from properties)
  - Nearest MRT (nearestmrt from **properties**, not rooms)
  - Rent (sellingprice from rooms)
  - Amenities (aircon, wifi, fridge, washer, dryer, gym, swimming, tenniscourt, squashcourt, microwave as Yes/No)
  - Available from (eavaildate from rooms)
  - Washroom details
4. **Syntactically Correct SQL**: Ensure the query is executable. Do not include explanatory text, comments, or non-SQL content in the query.
5. **Fuzzy Matching**: Use LIKE for MRT and location matches.
6. **Amenity Filtering**: If amenities are mentioned, filter rooms where the corresponding columns are 'true'.
7. **Budget Filtering**: If min_budget and max_budget are provided, filter rooms where sellingprice is within the range.
8. **Washroom Preference**: If washroom_preference is "private", filter rooms where washroomno = 1 and location = 'Within Room'. If "shared", filter rooms where washroomno > 1 or location != 'Within Room'.
9. **Follow-Up Questions**: If the user asks about specific room details (e.g., washroom details, amenities) after a room search, generate a query to fetch those details for the previously listed rooms. Use the buildingname, nearestmrt, and sellingprice from the last results to identify the rooms.
10. **Limit Results**: Limit to 5 rows unless specified otherwise.
11. **Aggregation for Non-Grouped Columns**: When using GROUP BY (e.g., on rooms.roomid), apply aggregate functions (e.g., MAX() or MIN()) to all non-grouped columns in the SELECT list (e.g., MAX(p.buildingname), MAX(p.nearestmrt), MAX(r.sellingprice)) to comply with SQL modes like ONLY_FULL_GROUP_BY.

SQL Query:
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

    # Combine conversation history into one string
    chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

    # Get user preferences from session state
    preferences = st.session_state["preferences"]

    # Construct the prompt
    prompt = f"""
    You are a SQL query generator. Generate a SQL query to retrieve rooms based on user preferences.

    User Preferences:
    {preferences}

    Generate a SQL query to retrieve rooms that match the user's preferences, including amenity columns (aircon, wifi, fridge, washer, dryer, gym, swimming, tenniscourt, squashcourt, microwave).
    Map the number of occupants to maxoccupancy. Filter by washroom preferences, preferred mrt, budget range.
    Filter for active rooms and properties (r.status = 'a' and p.status = 'a').
    Order the results by sellingprice in ascending order.
    Limit the results to the top 5.

    **Crucially, include the 'totalusers' and 'location' columns from the 'washrooms' table to describe the washroom details in the response.
    Specifically, indicate if the washroom is 'private' (totalusers = 1) or 'shared' (totalusers > 1), and provide the 'location' of the washroom.**

    Conversation History:
    {chat_history_str}

    User's Question:
    {user_question}

    Generate the SQL query:
    """

    response = sql_chain.invoke({
        "chat_history": chat_history_str,
        "question": prompt  # Pass the constructed prompt
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

def collect_customer_info():
    template = """
You are Aba from Adobha Co-living, assisting in a friendly and respectful manner.

I need to collect these details:
- Move-in date
- Rental duration
- Number of occupants
- Pass type (EP, S Pass, Work Permit, etc.)
- Work/study location
- Preferred MRT line
- Nationality
- Gender

Current preferences:
{preferences}

Conversation History:
{chat_history}

User’s latest query:
{question}

If the user is giving info, thank them and note it. If anything is missing, ask for it politely. If all details are collected, say: "Thank you! I’ve got everything I need. Let me find some great options for you!"

Aba’s response:
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-0125-preview")
    return (
        RunnablePassthrough.assign(preferences=lambda _: st.session_state.get("preferences", {}))
        | prompt
        | llm
        | StrOutputParser()
    )

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
You are Aba from Adobha Co-living. Collect the following from a new user:
- Rental duration (must be at least 3 months)
- Pass type (EP, Student, Work)
- Preferred MRT
- Washroom preference (private or shared)
- Number of occupants
- Move-in date
- Nationality
- Gender
- Budget range (min and max)
- Amenities (e.g., aircon, wifi, gym)

Current preferences:
{preferences}

Conversation History:
{chat_history}

User’s latest query:
{question}

If the user provides info, thank them and ask for the next missing piece of information.
If rental duration is less than 3 months, say: "We require a minimum rental duration of 3 months. Please provide a duration of at least 3 months."
If all info is collected, say: "Thank you! I’ve got everything I need. Let me find some great options for you!"
If the user asks a question or goes off-topic, address it briefly and then ask for the next missing piece.

Aba’s response:
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-0125-preview")
    return (
        RunnablePassthrough.assign(preferences=lambda _: st.session_state.get("preferences", {}))
        | prompt
        | llm
        | StrOutputParser()
    )

def generate_llm_response(chat_history, user_query, prompt_template):
    prompt = ChatPromptTemplate.from_template(prompt_template)
    llm = ChatOpenAI(model="gpt-4-0125-preview")
    chain = prompt | llm | StrOutputParser() #create chain.
    response = chain.invoke({"chat_history": chat_history, "user_query": user_query}) #invoke chain.
    return response #return string.


def classify_intent(chat_history, user_query):
    template = """
Classify the intent of the user's message. Return only one of the following options exactly: 
- general
- information_request
- property_search
- viewing_request
- error

Conversation History:
{chat_history}

User's Message:
{user_query}

Intent:
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-0125-preview")
    return prompt | llm | StrOutputParser()

# --- Helper to Check if All Info is Collected ---
def all_info_collected(preferences):
    required_fields = ['move_in_date', 'rental_duration', 'occupants', 'preferred_mrt', 'pass_type', 'work_study_location', 'nationality', 'gender']
    return all(field in preferences and preferences[field] for field in required_fields)

# --- Main Response Function ---
import logging
import streamlit as st
from langchain_core.messages import HumanMessage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_response(user_query: str, chat_history: list):
    new_history = chat_history + [HumanMessage(content=user_query)]
    
    # Initialize flag if not already set
    if "viewing_possible" not in st.session_state:
        st.session_state["viewing_possible"] = True  # Changed to True so viewing can be scheduled

    # Step 1: Determine user type if not set
    if st.session_state["user_type"] is None:
        logger.info(f"Determining user type for query: {user_query}")
        user_type = classify_user_type(chat_history, user_query)
        logger.info(f"Classified user type: {user_type}")
        if user_type == "existing":
            st.session_state["user_type"] = "existing"
            return "Thank you! Let's start by getting your address."
        elif user_type == "new":
            st.session_state["user_type"] = "new"
            return "Great! Let's find you a room. How long are you planning to rent?"
        else:
            return "I'm not sure if you're an existing or new user. Could you please clarify?"

    # Step 4: Check for viewing request FIRST before other handling
    # This is the key fix - we need to check for viewing request before other processing
    intent_chain = classify_intent(new_history, user_query)
    try:
        intent = intent_chain.invoke({"chat_history": new_history, "user_query": user_query})
        intent = intent.lower().strip()
        logger.info(f"Classified intent: {intent}")
        
        if intent == "viewing_request":
            viewing_prompt = """
You are Aba from Adobha Co-living. The user has requested a viewing. Respond appropriately by:

1. **Confirming Availability:**
    - Ask the user for their full name and phone number.
    - Validate the phone number to ensure it is a valid Singaporean number:
        - It should start with `+65` or `65`.
        - It should be followed by 8 digits (e.g., `+65 9123 4567` or `6591234567`).
    - If the phone number is invalid, politely ask the user to provide a valid Singaporean number.
    - Ask the user to specify their preferred date and time for the viewing.
    - If they have already provided a date and time, confirm that you have recorded it.
    - Once they provide the full details, display them for verification.

2. **Providing Room Address:**
    - Provide the complete address of the room they wish to view.

3. **Arrival Instructions:**
    - Ask the customer to notify you when they arrive at the location.
    - If applicable, mention who they should ask for or where they should go upon arrival.

4. **Expressing Gratitude:**
    - Thank the user for choosing Adobha Co-living.

Conversation History:
{chat_history}

User's Message:
{user_query}

Your Response:
"""
            response = generate_llm_response(new_history, user_query, viewing_prompt)
            return response
    except Exception as e:
        logger.error(f"Error classifying intent: {e}")
        # Continue with normal flow if intent classification fails

    # Step 2: Handle existing users
    if st.session_state["user_type"] == "existing":
        extracted_info = extract_existing_user_info(new_history)
        st.session_state["existing_user_info"].update(extracted_info)
        logger.info(f"Extracted existing user info: {st.session_state['existing_user_info']}")
        
        # Check if all info is collected
        if all(st.session_state["existing_user_info"].values()):
            return "Thank you! I've got everything I need. How can I assist with your problem?"
        
        # Ask for next missing piece
        chain = collect_existing_user_info_chain()
        response = chain.invoke({"question": user_query, "chat_history": chat_history})
        logger.info(f"Response for existing user: {response}")
        return response

    # Step 3: Handle new users (property search)
    if st.session_state["user_type"] == "new":
        # Extract preferences from chat history
        extracted_preferences = extract_preferences(new_history)
        st.session_state["preferences"].update(extracted_preferences)

        # Validate rental duration
        rental_duration = st.session_state["preferences"].get("rental_duration", "")
        if rental_duration and "month" in rental_duration.lower():
            try:
                months = int(rental_duration.split()[0])
                if months < 3:
                    st.session_state["preferences"]["rental_duration"] = None
                    return "We require a minimum rental duration of 3 months. Please provide a duration of at least 3 months."
            except ValueError:
                pass

        # Check if all preferences are collected for property search
        required_fields = ["rental_duration", "pass_type", "preferred_mrt", "washroom_preference",
                           "occupants", "move_in_date", "nationality", "gender", "min_budget", "max_budget"]
        if all(st.session_state["preferences"].get(field) for field in required_fields):
            sql_query = generate_query_from_preferences(new_history, user_query)
            results = execute_query(sql_query)
            if results["status"] == "success":
                # Store the results for later reference
                st.session_state["last_room_results"] = results["data"]
                chain = get_property_info_chain()
                return chain.invoke({"results": results["data"], "chat_history": chat_history, "question": user_query})
            elif results["status"] == "no_matches":
                return results["message"]
            else:
                return results["message"]

        # Ask for next missing piece for property search
        chain = collect_new_user_info_chain()
        return chain.invoke({"question": user_query, "chat_history": chat_history})

    # Handle other intents if not caught earlier
    try:
        if intent == "general" or intent == "casual_conversation":
            casual_prompt = """
You are Aba from Adobha Co-living, having a casual conversation with the user. Respond appropriately, maintaining a natural and friendly conversational flow.

If the user expresses gratitude, acknowledges your assistance, apologizes, or makes a similar conversational turn, respond appropriately and then seamlessly continue the conversation or ask how you can further assist them.

Conversation History:
{chat_history}

User's Message:
{user_query}

Your Response:
"""
            response = generate_llm_response(new_history, user_query, casual_prompt)
            return response

        elif intent == "information_request":
            info_prompt = """
You are Aba from Adobha Co-living. The user has requested information. Respond appropriately.

Conversation History:
{chat_history}

User's Message:
{user_query}

Your Response:
"""
            response = generate_llm_response(new_history, user_query, info_prompt)
            return response

        elif intent == "error":
            return "Sorry, I encountered an error while processing your request. Please try again or rephrase your query."
    except:
        pass

    # Default response if all else fails
    return "I'm not sure how to respond to that. Could you please clarify?"


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