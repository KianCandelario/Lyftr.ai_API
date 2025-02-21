import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Dict, Any
from supabase import create_client, Client
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, MessagesState, StateGraph
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import trim_messages
import uvicorn
from dotenv import load_dotenv

# Load the .env file from the root folder
env_path = os.path.join(os.path.dirname(__file__), "..", ".env")
load_dotenv(env_path)

# Set up Groq API key
if not os.environ.get("GROQ_API_KEY"):
    os.environ["GROQ_API_KEY"] = "groq-api-key"

# Initialize the chatbot model
model = init_chat_model("llama-3.3-70b-versatile", model_provider="groq")

# Define the prompt template
prompt_template = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "Act like a nutritionist, a physical therapist, or a gym instructor (depending on the type of question). Answer with the best of your ability."
        ),
        MessagesPlaceholder(variable_name="messages"),
    ]
)

# Define the message trimmer
trimmer = trim_messages(
    max_tokens=1024,  # For medium-length conversations
    strategy="last",  # Keeps the latest messages but may lose long-term contexts
    token_counter=model,
    include_system=True,
    allow_partial=False,
    start_on="human",
)

# Define the workflow
workflow = StateGraph(state_schema=MessagesState)

def call_model(state: MessagesState):
    trimmed_messages = trimmer.invoke(state["messages"])
    prompt = prompt_template.invoke({"messages": trimmed_messages})
    response = model.invoke(prompt)
    return {"messages": [response]}

workflow.add_edge(START, "model")
workflow.add_node("model", call_model)

memory = MemorySaver()
app_graph = workflow.compile(checkpointer=memory)

# Initialize Supabase client
supabase_url = os.environ.get("NEXT_PUBLIC_SUPABASE_URL")
supabase_key = os.environ.get("NEXT_PUBLIC_SUPABASE_ANON_KEY")
supabase: Client = create_client(supabase_url, supabase_key)

# Define a Pydantic model for the request body
class ChatRequest(BaseModel):
    user_id: Any
    prompt: str
    messages: List[Dict[str, Any]]  # List of previous messages

app = FastAPI()

# Enable CORS (Allows Next.js to make requests to FastAPI)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Chatbot endpoint
@app.post("/chatbot/response")
async def chatbot_response(chat_request: ChatRequest):
    try:
        config = {
            "configurable": {
                "thread_id": str(chat_request.user_id)
            }
        }
        # Convert messages to LangChain message format
        messages = []
        for msg in chat_request.messages:
            if msg["sender"] == "user":
                messages.append(HumanMessage(content=msg["content"]))
            else:
                messages.append(AIMessage(content=msg["content"]))

        # Add the new user prompt
        messages.append(HumanMessage(content=chat_request.prompt))

        # Invoke the chatbot workflow
        result = app_graph.invoke({"messages": messages}, config)

        # Extract the chatbot's response
        chatbot_response = result["messages"][-1].content

        # Save the user's prompt and chatbot's response to the message table
        supabase.table("messages").insert([
            {"user_id": chat_request.user_id, "sender": "user", "content": chat_request.prompt},
            {"user_id": chat_request.user_id, "sender": "bot", "content": chatbot_response},
        ]).execute()

        # Return the response
        return JSONResponse(content={"response": chatbot_response})
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Get Render's assigned port
    uvicorn.run(app, host="0.0.0.0", port=port)