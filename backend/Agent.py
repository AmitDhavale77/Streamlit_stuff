from .AutogenWrappers import find_predictions_wrapper, build_profiles_wrapper, verify_prediction_wrapper, calculate_credibility_scores_batch_wrapper
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_core import CancellationToken
import os 
from dotenv import load_dotenv
# dotenv_path = '/etc/secrets/.env'
# loaded = load_dotenv(dotenv_path=dotenv_path)
# if not loaded:
#      # Fallback in case it's mounted at root instead
#      load_dotenv()

# Initialize the API keys and URLs
OPEN_AI_KEY = os.environ.get("OPEN_AI_KEY")

MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-2024-08-06")

client1  = OpenAIChatCompletionClient(
    model = MODEL_NAME,
    # base_url="https://api.openai.com/v1",
    api_key=OPEN_AI_KEY,
)

assistant = AssistantAgent(
    name="SwarmcentsHelper",
    # llm_config=llm_config,
    model_client=client1,
    tools=[find_predictions_wrapper, build_profiles_wrapper, verify_prediction_wrapper, calculate_credibility_scores_batch_wrapper],
    system_message="""You are a prediction analysis expert that helps users find, profile, and verify predictions.
    
STRICT RULES YOU MUST FOLLOW:
1. You MUST ONLY use the provided functions - never make up data or predictions
2. You MUST ask clarifying questions if the user request is unclear
3. You MUST verify all predictions before making claims about accuracy
4. You MUST direct the user to choose one of the four function options

AVAILABLE FUNCTIONS:
1. find_predictions(user_prompt) - Finds posts containing predictions on a topic 
2. build_profile(handle) - Builds a profile of a predictor based on their history
3. verify_prediction(prediction_query) - Verifies if a prediction came true
4. calculate_credibility(handle) - Calculates a credibility score for a predictor

Respond ONLY with one of these approaches:
- Ask clarifying questions if needed
- Propose which function to use based on the user's need
- Execute one of the functions with appropriate parameters
- Present results from function calls (never make up data)
""",
reflect_on_tool_use=True,
model_client_stream=True,  # Enable streaming tokens from the model client.
)

async def run_prediction_analysis(text_messages):

    response = await assistant.on_messages(text_messages,
        cancellation_token=CancellationToken(),
    )
    print(response.inner_messages)
    print(response.chat_message)
    return response.chat_message.content