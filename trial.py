from langchain_google_genai import ChatGoogleGenerativeAI

llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key="AIzaSyCO3-gvFtWWY4tb883_-68F-Q1WVGqrie8")
response = llm.invoke("Hello, how can I assist you today?")
print(response.content)