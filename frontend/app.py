import streamlit as st
from autogen_ext.models.openai import OpenAIChatCompletionClient
import os
import os
import asyncio
from openai import OpenAI
import warnings
from autogen_agentchat.messages import TextMessage
import sys 
from dotenv import load_dotenv
# dotenv_path = '"C:\Amit_Laptop_backup\Imperial_essentials\AI Society\Hackathon Torus\.env"'
# loaded = load_dotenv(dotenv_path=dotenv_path)
# if not loaded:
#      # Fallback in case it's mounted at root instead
#      load_dotenv()
# --- Configuration ---
MAX_HISTORY_TURNS = 5 # Keep last 5 pairs (user+assistant) for context

os.environ["AUTOGEN_DEBUG"] = "0"  # Basic debug info
os.environ["AUTOGEN_VERBOSE"] = "0"  # More detailed logging

# import toml
# import os

# # Load secrets from secrets.toml
# secrets = toml.load("C:\Amit_Laptop_backup\Imperial_essentials\AI Society\Hackathon Torus\secrets.toml")

# # Set environment variables
# for key, value in secrets.items():
#     os.environ[key] = value

# Add the project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Add this near the top of your script
warnings.filterwarnings("ignore", message=r"Model .* is not found. The cost will be 0.*")

# --- Streamlit App ---
st.set_page_config(page_title="SwarmCents Chat", page_icon="💰", layout="wide")
st.markdown("""
    <h1 style='text-align: center; color: gold;'>
        💰 SwarmCents Chat
    </h1>
    <p style='text-align: center; color: lightgray; font-size: 16px;'>
        Your AI guide to Polymarket predictions 📊🔍
    </p>
""", unsafe_allow_html=True)

# API Keys
OPEN_AI_KEY = os.environ.get("OPEN_AI_KEY")
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-2024-08-06")

# Sidebar
with st.sidebar:
    st.markdown("## 🛠️ How to use")
    st.markdown("1. Choose your Model 🤖\n"
                "2. Ask a question through natural language. 💬")

    st.markdown("---")
    st.markdown("### ⚙️ API Configuration")
    selected_model = st.selectbox("Choose Model", ["gpt-4o", "grok 2", "o3-mini"], index=0)
    
    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("💰 <b>SwarmCents Chat</b> helps you explore and analyze predictions on Polymarket using AI.", unsafe_allow_html=True)

# Check for valid API Key & Model
if not OPEN_AI_KEY or not selected_model:
    st.warning("You must provide a valid OpenAI API key and select a model!", icon="⚠️")
    st.stop()

client = OpenAI(
    base_url="https://api.openai.com/v1",
    api_key=OPEN_AI_KEY
)
client1  = OpenAIChatCompletionClient(
    model = MODEL_NAME,
    # base_url="https://api.openai.com/v1",
    api_key=OPEN_AI_KEY,
)

from backend.Agent import run_prediction_analysis

# --- Initialize Chat History in Session State ---
INITIAL_MESSAGE = [
    {
        "role": "assistant",
        "content": 
"""Hey there, I'm SwarmCents Chat! I'm here to help you analyze predictions about Polymarket topics made on Twitter.
Please select one of these specific options:
1. Find predictions on [specific topic], also give account names of users who made them. 
2. Build profile for [@predictor_handle]. Show me their all prediction history and analysis.
3. Verify prediction: "[exact prediction text]"
4. Calculate the credibility score for [@predictor_handle].
Which option would you like to proceed with?""",
    },
]

# Add a reset button
if st.sidebar.button("🔄 Reset Chat"):
    for key in st.session_state.keys():
        del st.session_state[key]
    # Re-initialize after deleting
    st.session_state.messages = list(INITIAL_MESSAGE) # Ensure it's a mutable list copy
    st.rerun() # Force rerun after reset


# Display the chat messages
if "messages" not in st.session_state:
    st.session_state.messages = INITIAL_MESSAGE

def run_async_function(coro):
    try:
        loop = asyncio.get_running_loop()
        task = loop.create_task(coro)
        return loop.run_until_complete(task)
    except RuntimeError:
        # No running loop, so create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(coro)
        finally:
            # Clean up any lingering async generators
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.close()

# Initialize chat messages in session state    
if "messages" not in st.session_state:
    st.session_state.messages = list(INITIAL_MESSAGE) # Use list() to ensure mutable copy

# --- Display Chat History ---
avatar_assistant = None
avatar_user = None
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar=avatar_assistant if message["role"] == "assistant" else avatar_user):
        st.markdown(message["content"])

# --- Add a flag in session state to prevent double input ---
if "is_waiting" not in st.session_state:
    st.session_state.is_waiting = False

# --- Handle User Input ---
prompt_disabled = st.session_state.is_waiting  # Disable input if waiting for a response
prompt = st.chat_input("Type your Query", disabled=prompt_disabled)

# --- Handle User Input ---
if prompt:
    # --- Set waiting flag to True ---
    st.session_state.is_waiting = True

    # 1. Append user message to FULL history (for display)
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar=avatar_user):
        st.markdown(prompt)

    # 2. Prepare the LIMITED history to send to the agent
    # Keep only the last N messages (user + assistant pairs)
    # Multiply by 2 for pairs, add 1 if you always want the initial system message if any
    history_limit = MAX_HISTORY_TURNS * 2
    # limited_history = st.session_state.messages[-history_limit:]
    limited_history = []
    for m in st.session_state.messages[-history_limit:]:
        if m.get("role") == "tool":
            # Break it by emptying the content
            broken_tool_msg = m.copy()
            broken_tool_msg["content"] = ""  # or `None` if you want to test that too
            limited_history.append(broken_tool_msg)
        else:
            limited_history.append(m)
    print("Limited History:", limited_history)
    # Convert the LIMITED history to the format expected by your agent
    text_messages_for_agent = [
        # Assuming TextMessage takes content and source)
        TextMessage(content=m["content"], source=m["role"]) # Adjust 'role' if it expects 'source' etc.
        for m in limited_history
    ]

    # 3. Call the agent with the LIMITED history
    with st.chat_message("assistant", avatar=avatar_assistant):
        placeholder = st.empty()
        placeholder.markdown("Thinking...")
        try:
            # Pass the truncated message list to your backend
            response = run_async_function(run_prediction_analysis(text_messages_for_agent))
            placeholder.markdown(response)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            response = "Sorry, I encountered an error." # Provide a fallback response
            placeholder.markdown(response)
            print("AN EXCEPTION OCCURED", e)
            # st.session_state.clear() Genev changes
            # st.rerun() Genev changes


    # 4. Append assistant response to FULL history (for display)
    st.session_state.messages.append({"role": "assistant", "content": response})

    # 5. Optional: Rerun to ensure the latest message is displayed immediately if needed
    # st.rerun() # Usually not needed as Streamlit handles updates, but can force it.

    # --- Reset waiting flag ---
    st.session_state.is_waiting = False
