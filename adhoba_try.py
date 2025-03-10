import os
import streamlit as st
import pandas as pd
import json
import re
from pathlib import Path
from dotenv import load_dotenv
import sqlparse

from langchain_community.utilities.sql_database import SQLDatabase
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
st.set_page_config(page_title="Adobha Co-living Assistant", page_icon="🏠")
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
    st.session_state["chat_history"] = [AIMessage(content="Hi there! I’m Justine from Adobha Co-living. How can I assist you today?")]
if "preferences" not in st.session_state:
    st.session_state["preferences"] = {}
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

# --- Extraction of Preferences ---
def extract_preferences(chat_history):
    """Extract preferences from chat history."""
    chat_history_str = "\n".join([f"{msg.type}: {msg.content}" for msg in chat_history])

    template = """
You are an assistant that extracts user preferences from a conversation. Extract the following fields exactly as provided:

- move_in_date: The user's preferred move-in date (e.g., "May 1st").
- rental_duration: The duration of the rental (e.g., "1 year").
- occupants: The number of occupants (e.g., "1").
- preferred_mrt: The user's preferred MRT line (e.g., "Simei MRT"). If the MRT line is mentioned anywhere in the conversation (even if not explicitly labeled), extract it. If multiple MRT lines are mentioned, use the most recent valid one provided.
- pass_type: The user's pass type (e.g., "EP", "S Pass", "Work Permit").
- work_study_location: The user's work or study location (e.g., "CBD", "NUS").
- nationality: The user's nationality (e.g., "Singaporean", "American").
- gender: The user's gender (e.g., "male", "female", "other").

If a field is not mentioned or cannot be determined, set its value to null.

Return the extracted preferences as a JSON object with the following structure:
{{
    "move_in_date": "value or null",
    "rental_duration": "value or null",
    "occupants": "value or null",
    "preferred_mrt": "value or null",
    "pass_type": "value or null",
    "work_study_location": "value or null",
    "nationality": "value or null",
    "gender": "value or null"
}}

Conversation History:
{chat_history}

Extracted Preferences:
"""
    llm = ChatOpenAI(model="gpt-4-0125-preview")
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | llm | StrOutputParser()

    response = chain.invoke({"chat_history": chat_history_str})
    st.write("LLM Raw Response:", response)
    cleaned_response = clean_response(response)
    st.write("Cleaned Response:", cleaned_response)

    try:
        preferences = json.loads(cleaned_response)
    except json.JSONDecodeError:
        st.error("Failed to parse LLM response as JSON.")
        preferences = {}

    preferences = fallback_mrt(chat_history, preferences)
    return preferences

# --- Dynamic SQL Query Generation using LLM ---
def get_sql_chain(db):
    template = """
    You are a data analyst at Adobha Co-living. Based on the database schema provided and the conversation history, generate a valid SQL query that finds properties matching the user's preferences.

Database Schema:
{schema}

Conversation History: {chat_history}

Question: {question}

**Important Instructions:**

1.  **Date Handling:** Always use the `DATE()` function to extract the date part from datetime columns before comparing them with date values. This is crucial for accurate date comparisons.
2.  **Status Handling:** The room `status` field indicates availability with the value 'a'. Only include rooms where `rooms.status = 'a'`. The properties table `status` field also indicates availability with the value 'a'. Include only rows where `properties.status = 'a'`.
3.  **Property Details:** When describing properties, include the following details in your query results:
    * Price (`sellingprice`)
    * Location (`add1` from properties table)
    * Availability date (`eavaildate`)
    * Tenant mix (`ptenanttype`)
    * Proximity to MRT (`nearestmrt` from rooms table)
    * Amenities (aircon, wifi, fridge, washer, dryer, gym, swimming, tenniscourt, squashcourt, microwave as Yes or No)
    * Nearest Supermarket (`nearestsupermarket` from properties table)
    * Nearest Food Court (`nearestfoodcourt` from properties table)
    * Nearest Bus Stop (`nearestbusstop` from properties table)
    * Property Type (`propertytype` from properties table)
    * Building Name (`buildingname` from properties table)
    * Washroom Details (washroomno, size, bathtub, location, bidetspray, totalusers, usedrooms from washrooms table, using aggregate functions to combine multiple washrooms into single row)
4.  **Syntactically Correct SQL:** Ensure the generated SQL query is syntactically correct and executable.
5.  **No Assumptions:** Do not assume any additional details beyond what is provided in the schema and conversation history.
6.  **Avoid Redundant Date Checks:** Do not include date checks that compare the current date with past dates, unless explicitly requested by the user.
7.  **Boolean Formatting:** Display boolean values (aircon, wifi, bathtub, bidetspray, etc.) as 'Yes' or 'No' in the result.
8.  **Fuzzy Matching:** When the user mentions a location or MRT station, use the `LIKE` operator with wildcards (`%`) to find matching entries. Do not require exact matches.
9.  **Price Column:** The price of the room is now represented by the `sellingprice` column in the rooms table. Use this column for price related queries.
10. **Do not use the `published` column in the query.**
11. **Use the `nearestmrt` column from the rooms table to find the nearest mrt. Do not use the `mrt` column from the properties table.**
12. **Include washroom details from the washrooms table in the query results. Use aggregate functions like `GROUP_CONCAT()` to combine multiple washrooms into a single row.**
13. **Data Quality:** The `rooms.status` column contains inconsistent data. For this query, only use rows where `rooms.status` is exactly equal to 'a'. Do not use rows where `rooms.status` is 'i' or an empty string.
14. **Limit Results:** If the user does not specify a limit, limit the results to a maximum of 5 rows to prevent overwhelming the user.
15. **Amenity Filtering:** If the user mentions any amenity (aircon, wifi, fridge, washer, dryer, gym, swimming, tenniscourt, squashcourt, microwave) in their query, filter the results to include only rooms where the corresponding column in the `rooms` table is equal to 'true'.

    SQL Query:
    """
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-0125-preview")

    def get_schema(_):
        return db.get_table_info()

    return (
        RunnablePassthrough.assign(schema=get_schema)
        | prompt
        | llm
        | StrOutputParser()
    )

def clean_sql_query(query: str) -> str:
    """Remove markdown code fences and any language identifier from the SQL query."""
    query = query.strip()
    query = re.sub(r"^```(?:sql)?\s*", "", query)
    query = re.sub(r"\s*```$", "", query)
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
    response = sql_chain.invoke({
        "chat_history": chat_history_str,
        "question": user_question
    })
    # Strip and clean the response
    raw_query = response.strip()
    sql_query = clean_sql_query(raw_query)
    return sql_query


def execute_query(query: str):
    try:
        # Create a direct connection
        connection = engine.raw_connection()
        try:
            # Create a cursor
            cursor = connection.cursor()
            try:
                # Execute the query
                cursor.execute(query)
                # Fetch all results
                columns = [col[0] for col in cursor.description]
                rows = cursor.fetchall()
                # Create a DataFrame from the results
                df = pd.DataFrame(rows, columns=columns)
                
                if df.empty:
                    return {"status": "no_matches", "message": "I couldn't find any properties matching your preferences. Want to tweak them a bit?"}
                return {"status": "success", "data": df.to_dict(orient="records")}
            finally:
                # Close the cursor
                cursor.close()
        finally:
            # Close the connection
            connection.close()
    except Exception as e:
        return {"status": "error", "message": f"Oops, something went wrong with the query: {e}"}


# --- Response Chains for Final Answer ---
def get_property_info_chain():
    template = """
    You are Justine from Adobha Co-living, here to assist in a friendly and respectful way.

The user is looking for available **rooms** for rent, and here’s what I found:

<RESULTS>
{results}
</RESULTS>

Conversation History:
{chat_history}

User’s latest query:
{question}

Please share a helpful and detailed response with the **room** details, keeping it clear and concise. 
Highlight key features or amenities of the **rooms** that match the user's previously stated preferences, if possible.
Use bullet points or numbered lists for clarity.
If there are no matching **rooms**, inform the user and suggest alternative options.
If there are multiple options, list the **rooms** briefly.
Do not mention the <RESULTS> tags in your response.

**Room Details:**
Please provide a summary of the room, including the location, price, availability date, tenant mix, and key amenities (e.g., air conditioning, proximity to MRT).

**Washroom Details:**
The washroom details are provided in the "WashroomDetails" column of the provided data.
Each entry represents a washroom and is formatted as follows:
"washroomno|size|bathtub|location|bidetspray|totalusers|usedrooms"
Multiple washrooms are separated by "; ".
Please parse these details and present them in a clear, readable format, reporting the details exactly as they appear.
Do not add any extra information or invent any details that are not present in the "WashroomDetails" column.
Keep the washroom details concise and focus on the most relevant information.

Example:
Input: "Juniorbath|Standard|No|Within Room|No|1|80; MasterBath|Large|No|Within Room|No|1|79"
Output:
- Juniorbath:
    - Size: Standard
    - Bathtub: No
    - Location: Within Room
    - Bidet Spray: No
    - Total Users: 1
    - Used Rooms: 80
- MasterBath:
    - Size: Large
    - Bathtub: No
    - Location: Within Room
    - Bidet Spray: No
    - Total Users: 1
    - Used Rooms: 79

If the "WashroomDetails" column is empty or null, state "No washroom details available."

Ask the user if they would like to schedule a viewing of any of the **rooms** or if they have any further questions.

Justine’s response:
"""
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-0125-preview")
    return prompt | llm | StrOutputParser()

def collect_customer_info():
    template = """
You are Justine from Adobha Co-living, assisting in a friendly and respectful manner.

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

Justine’s response:
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
Classify the intent of the user's message.

Conversation History:
{chat_history}

User's Message:
{user_query}

Intent (general, information_request, property_search, viewing_request, etc.):
"""
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-0125-preview")
    return prompt | llm | StrOutputParser()

# --- Helper to Check if All Info is Collected ---
def all_info_collected(preferences):
    required_fields = ['move_in_date', 'rental_duration', 'occupants', 'preferred_mrt', 'pass_type', 'work_study_location', 'nationality', 'gender']
    return all(field in preferences and preferences[field] for field in required_fields)

# --- Main Response Function ---
def get_response(user_query: str, chat_history: list):
    new_history = chat_history + [HumanMessage(content=user_query)]

    intent_sequence = classify_intent(chat_history, user_query)
    try:
        intent = intent_sequence.invoke({"chat_history": chat_history, "user_query": user_query})
    except Exception as e:
        st.write(f"Error classifying intent: {e}")
        return "Sorry, I encountered an error. Please try again."

    intent = intent.lower()

    if "general" in intent or "casual_conversation" in intent:
        casual_prompt = """
You are Justine from Adobha Co-living, having a casual conversation with the user. Respond appropriately, maintaining a natural and friendly conversational flow.

If the user expresses gratitude, acknowledges your assistance, apologizes, or makes a similar conversational turn, respond appropriately and then seamlessly continue the conversation or ask how you can further assist them.

Conversation History:
{chat_history}

User's Message:
{user_query}

Your Response:
"""
        response = generate_llm_response(new_history, user_query, casual_prompt)
        return response

    elif "property_search" in intent:
        property_prompt = """
You are Justine from Adobha Co-living, assisting a customer with a property search. Respond appropriately.

Conversation History:
{chat_history}

User's Message:
{user_query}

Your Response:
"""
        new_preferences = extract_preferences(new_history)
        st.session_state["preferences"] = {**st.session_state.get("preferences", {}), **new_preferences}
        st.write("Merged Preferences:", st.session_state["preferences"])
        save_preferences_to_file(st.session_state["preferences"])
        st.write("Preferences saved to JSON file.")
        current_preferences = st.session_state["preferences"]

        if not all_info_collected(current_preferences):
            st.write("All Info Collected: False")
            st.session_state["collecting_info"] = True
            chain = collect_customer_info()
            response = chain.invoke({"question": user_query, "chat_history": chat_history})
            return response

        st.write("All Info Collected: True")
        st.session_state["collecting_info"] = False

        try:
            sql_query = generate_query_from_preferences(new_history, user_query)
            st.write("Generated SQL Query:", sql_query)
            results = execute_query(sql_query)

            if results["status"] == "success":
                chain = get_property_info_chain()
                response = chain.invoke({
                    "results": results["data"],
                    "chat_history": chat_history,
                    "question": user_query
                })
                return response
            elif results["status"] == "no_matches":
                return results["message"]
            else:
                return results["message"]

        except Exception as e:
            st.write(f"Error during query execution: {e}")
            return "An error occurred while processing your request."

    elif "viewing_request" in intent:
        viewing_prompt = """
You are Justine from Adobha Co-living. The user has requested a viewing. Respond appropriately by:

1. **Confirming Availability:**
    - Ask the user to specify their preferred date and time for the viewing.
    - If they have already provided a date and time, confirm that you have recorded it.

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

    elif "information_request" in intent:
        info_prompt = """
You are Justine from Adobha Co-living. The user has requested information. Respond appropriately.

Conversation History:
{chat_history}

User's Message:
{user_query}

Your Response:
"""
        response = generate_llm_response(new_history, user_query, info_prompt)
        return response

    elif "error" in intent:
        return "Sorry, I encountered an error while processing your request. Please try again or rephrase your query."

    else:
        return "I'm not sure how to respond to that. Could you please clarify?"
# --- UI Setup ---
st.title("GenZI Care Chat Bot")

with st.sidebar:
    st.subheader("About Adobha Co-living")
    st.write("Singapore’s pioneer in co-living since 2013.")
    st.subheader("Our Offerings")
    st.write("• We provide a variety of fully furnished rooms with flexible lease terms and all-inclusive pricing.")
    st.subheader("Developed by GenZI care")
    if st.button("Reset Chat"):
        st.session_state["chat_history"] = [AIMessage(content="Hi there! I’m Justine from Adobha Co-living. How can I assist you today?")]
        st.session_state["preferences"] = {}
        st.session_state["collecting_info"] = False
        st.rerun()

# --- Chat UI ---
for message in st.session_state["chat_history"]:
    if isinstance(message, AIMessage):
        with st.chat_message("AI", avatar="👩‍💼"):
            st.markdown(message.content)
    elif isinstance(message, HumanMessage):
        with st.chat_message("Human", avatar="👤"):
            st.markdown(message.content)

user_query = st.chat_input("Drop your message here...")
if user_query and user_query.strip():
    st.session_state["chat_history"].append(HumanMessage(content=user_query))
    with st.chat_message("Human", avatar="👤"):
        st.markdown(user_query)
    with st.chat_message("AI", avatar="👩‍💼"):
        response = get_response(user_query, st.session_state["chat_history"])
        st.markdown(response)
    st.session_state["chat_history"].append(AIMessage(content=response))
