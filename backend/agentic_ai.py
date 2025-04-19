import os
import json
import requests
import re
from typing import List, Dict, Tuple
from autogen import AssistantAgent
import os
import asyncio
from openai import OpenAI
os.environ["AUTOGEN_DEBUG"] = "0"  # Basic debug info
os.environ["AUTOGEN_VERBOSE"] = "0"  # More detailed logging1
import warnings
import re
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.messages import TextMessage
from autogen_core import CancellationToken
from autogen_ext.models.openai import OpenAIChatCompletionClient
from dotenv import load_dotenv
import os
import time
from datura_py import Datura

# Path where Render mounts the secret file
dotenv_path = "C:\Amit_Laptop_backup\Imperial_essentials\AI Society\Hackathon Torus\.env"
loaded = load_dotenv(dotenv_path=dotenv_path)
if not loaded:
     # Fallback in case it's mounted at root instead
     load_dotenv()
# Add this near the top of your script
load_dotenv()
warnings.filterwarnings("ignore", message=r"Model .* is not found. The cost will be 0.*")

# API Keys - Replace with your actual keys
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY1 = os.getenv("GROQ_API_KEY")
GROQ_API_KEY2 = os.getenv("GROQ_API_KEY")

DATURA_API_KEY = os.getenv("DATURA_API_KEY")
NEWS_API_TOKEN = os.getenv("NEWS_API_TOKEN")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GOOGLE_CSE_ID = os.getenv("GOOGLE_CSE_ID")

OPEN_AI_KEY = os.getenv("OPEN_AI_KEY")

MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4o-2024-08-06")
MODEL_NAME1 = os.getenv("MODEL_NAME", "gpt-4o-mini-2024-07-18")  
DATURA_API_URL = "https://apis.datura.ai/twitter"

client = OpenAI(
    base_url="https://api.openai.com/v1",
    api_key=OPEN_AI_KEY,
)
client1  = OpenAIChatCompletionClient(
    model = MODEL_NAME,
    # base_url="https://api.openai.com/v1",
    api_key=OPEN_AI_KEY,
)


# ============ COMPONENT 1: PREDICTION FINDER ============

class PredictionFinder:
    """Finds tweets containing predictions about specified topics."""
    
    def __init__(self, groq_client, datura_api_key, datura_api_url):
        self.groq_client = groq_client
        self.datura_api_key = datura_api_key
        self.datura_api_url = datura_api_url
    
    def generate_search_query(self, user_prompt: str) -> str:
        """Generate a properly formatted search query from user prompt."""
        context = """You are an expert in constructing search queries for the Datura API to find relevant tweets related to Polymarket predictions.
Your task is to generate properly formatted queries based on user prompts.

Here are some examples of well-structured Datura API queries:

1. I want mentions of predictions about the 2024 US Presidential Election, excluding Retweets, with at least 20 likes.  
   Query: (President) (elections) (USA) min_faves:20  

2. I want tweets from @Polymarket users discussing cryptocurrency price predictions, excluding tweets without links.  
   Query: (Bitcoin) (Ethereum) (crypto) 

3. I want tweets predicting the outcome of the Wisconsin Supreme Court election between Susan Crawford and Brad Schimel.  
   Query: (Wisconsin) (SupremeCourt) (Crawford) (Schimel) (election)

4. I want tweets discussing AI stock price predictions in 2025  
   Query: (AI) (tech) (stock)

5. I want mentions of predictions about the winner of the 2025 NCAA Tournament.  
   Query: (NCAA) (MarchMadness) (2025) (winner) 

6. I want tweets discussing whether Yoon will be out as president of South Korea before May.  
   Query: (Yoon) (SouthKorea) (president) (resign) (before May) 

Now, given the following user prompt, generate a properly formatted Datura API query. (Just the query, no additional text or explanation.)"""

        completion = self.groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": user_prompt}
            ]
        )
        
        return completion.choices[0].message.content.strip()

    async def get_tweets(self, user_prompt: str, min_likes: int = 0, count: int = 100, max_retries = 5) -> List[Dict]:
        #Fetch tweets from Datura API based on the user prompt.

        payload = {
            "prompt": user_prompt,
            "model": "HORIZON",
            "start_date": "2024-04-10",
            "lang": "en",
            "verified": False,
            "blue_verified": False,
            "is_quote": False,
            "is_video": False,
            "is_image": False,
            "min_retweets": 0,
            "min_replies": 0,
            "min_likes": min_likes,
            "count": count
        }
        
        headers = {
            "Authorization": self.datura_api_key,
            "Content-Type": "application/json"
        }
        
        for attempt in range(max_retries):
            try:
                #print(f"🔁 Attempt {attempt + 1} to fetch tweets...")
                response = await asyncio.to_thread(requests.post, url=self.datura_api_url, json=payload, headers=headers)
                response.raise_for_status()
                data = response.json()
                tweets_ls = data.get("miner_tweets", [])
                #print(tweets_ls)
                #print(len(tweets_ls), "tweets found")
                if len(tweets_ls) > 0:
                    return tweets_ls
                
            except requests.exceptions.RequestException as e:
                print(f"❌ Attempt {attempt + 1} failed: {e}")
                if attempt == max_retries-1:
                    print("🚫 Max retries reached. Returning error.")
                    return []
            
            print(f"Attempt {attempt + 1} failed. Retrying...")
            await asyncio.sleep(2)
        
        print("🚫 Max retries reached. Returning error.")
        return {"error": "Invalid Username. No tweets found after 5 attempts.", "tweets": []}

    def process_tweets(self, tweets: List[Dict]) -> Tuple[Dict, Dict]:
        """Process tweets to create structured data."""
        hash_dict = {}
        username_to_tweet = {}
        
        for tweet in tweets:
            user_id = tweet["user"]["id"]
            
            hash_dict[user_id] = {
                "username": tweet["user"]["username"],
                "favourites_count": tweet["user"]["favourites_count"],
                "is_blue_verified": tweet["user"]["is_blue_verified"],
                "tweet_text": tweet["text"],
                "like_count": tweet["like_count"],
                "created_at": tweet["created_at"],
                "tweet url": tweet["url"],
            }
            
            username_to_tweet[tweet["user"]["username"]] = tweet["text"]
        
        return hash_dict, username_to_tweet
    
    def analyze_predictions(self, username_to_tweet: Dict) -> str:
        """Analyze tweets to identify predictions."""
        json_string = json.dumps(username_to_tweet, indent=4)
        
        context = """You are an expert in identifying explicit and implicit predictions in tweets related to Polymarket topics.

Here is a JSON object containing tweets, where each key represents a unique tweet ID and the value is the tweet text.

Your task:  
For each tweet, determine if it contains an **explicit or implicit prediction** about a future event **related to a Polymarket topic**.  
- If it **does**, return "Yes".  
- If it **does not**, return "No".  

Format your response as a JSON object with each tweet ID mapped to "Yes" or "No".

Example:

Input JSON:
{
    "username1": "Bitcoin will hit $100K by the end of 2025!",
    "username2": "The economy is in trouble. People are struggling.",
    "username3": "I bet Trump wins the next election."
}

Expected Output:
{
    "username1": "Yes",
    "username2": "No",
    "username3": "Yes"
}

Now, analyze the following tweets and generate the output: 
Ensure the response is **valid JSON** with no additional text.
"""
        
        completion = self.groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": json_string}
            ]
        )
        
        return completion.choices[0].message.content
    
    def filter_tweets_by_prediction(self, yes_no: str, hash_dict: Dict) -> str:
        """Filter tweets to only include those with predictions."""
        match_yes_no = re.search(r"\{(.*)\}", yes_no, re.DOTALL)
        json_content_yes_no = "{" + match_yes_no.group(1) + "}"
        
        yes_no_dict = json.loads(json_content_yes_no)
        
        filtered_tweets = {
            tweet_id: details
            for tweet_id, details in hash_dict.items()
            if details["username"] in yes_no_dict and yes_no_dict[details["username"]] == "Yes"
        }
        
        return json.dumps(filtered_tweets, indent=4)
    
    async def find_predictions(self, user_prompt: str) -> Dict:
        """Main method to find predictions based on user prompt."""

        print(f"Generated Search Query: {user_prompt}")
        
        # Get tweets
        tweets = await self.get_tweets(user_prompt)
        
        if not tweets:
            return {"error": "No tweets found matching the criteria"}

        # Process tweets
        hash_dict, username_to_tweet = self.process_tweets(tweets)

        # Analyze predictions
        prediction_analysis = self.analyze_predictions(username_to_tweet)

        # Filter tweets
        filtered_predictions = self.filter_tweets_by_prediction(prediction_analysis, hash_dict)

        # Return as dictionary
        return json.loads(filtered_predictions)

# ============ COMPONENT 3: PREDICTOR VERIFIER ============

class PredictionVerifier:
    """Verifies whether predictions have come true or proven false."""
    
    def __init__(self, groq_client, news_api_token, google_api_key, google_cse_id):
        self.groq_client = groq_client
        self.news_api_token = news_api_token
        self.google_api_key = google_api_key
        self.google_cse_id = google_cse_id
        self.datura = Datura(api_key=DATURA_API_KEY)
    
    def fetch_google_results(self, query: str) -> List[Dict]:
        """Fetch search results from Google Custom Search API."""
        google_url = f"https://www.googleapis.com/customsearch/v1?q={query}&key={self.google_api_key}&cx={self.google_cse_id}&num=3"
        
        response = requests.get(google_url)
        # print("Google URL", response)
        if response.status_code == 200:
            data = response.json()
            # print("Google data", data)
            print("Capturing the data", data.get("data", []))
            return data.get("items", [])[:2]
        
        return []

    def generate_search_query(self, prediction_query: str) -> str:
        """Generate a concise question-style search query from a multi-paragraph prediction tweet."""
        context = """
        You are an expert at analyzing long prediction tweets (2-3 paragraphs) and extracting the core prediction to create concise, question-style search queries for Perplexica.

        Guidelines:
        1. Read the entire tweet carefully, focusing on the main prediction
        2. Identify the key subject, event, and timeframe
        3. Ignore supporting arguments or explanations
        4. Convert the core prediction into a natural-sounding question
        5. Keep it under 15 words when possible

        Examples:
        1. Prediction tweet: 'After analyzing market trends and political indicators, I believe there's a 52% chance that the UK will vote to leave the European Union in the 2016 referendum. This accounts for... [2 more paragraphs]'
        Query: What were the chances of Brexit happening in 2016?

        2. Prediction tweet: 'Considering current polling data and historical trends, my model shows a 30% probability that Donald Trump could win the 2016 US Presidential Election. Factors include... [3 paragraphs]'
        Query: Was Trump likely to win the 2016 election?

        3. Prediction tweet: 'Based on early adoption rates and technology reviews, there's an 80% probability that Apple's iPhone will revolutionize the smartphone industry when it launches in 2007. [2 more paragraphs explaining]'
        Query: Did experts predict iPhone's success in 2007?

        4. Prediction tweet: 'After evaluating team performance and tournament statistics, I estimate India has a 52% chance of winning the T20 Cricket World Cup Final in 2024. The analysis shows... [3 paragraphs]'
        Query: Were India favorites for the 2024 T20 World Cup?

        5. Prediction tweet: 'Cryptocurrency volatility patterns suggest a 40% probability Bitcoin could reach $100,000 by December 2021. My model accounts for... [2 paragraphs of technical analysis]'
        Query: Could Bitcoin hit $100k in 2021?

        Now generate a concise question query (only the question, no extra text) for this prediction tweet:
        """
        
        completion = self.groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": context},
                {"role": "user", "content": prediction_query},
            ],
        )
        
        return completion.choices[0].message.content.strip()

    def fetch_news_articles(self, search_query: str) -> List[Dict]:
        """Fetch news articles related to the prediction, with up to 5 retries."""

        max_retries = 5
        delay = 1  # Start with 1 second delay

        for attempt in range(1, max_retries + 1):
            try:
                result = self.datura.basic_web_search(
                    query=search_query,
                    num=5,
                    start=1
                )

                data = result.get("data", [])

                print(f"Attempt {attempt}: Captured data ->", data)

                if data:  # If we got results, return them
                    return data

            except Exception as e:
                print(f"Attempt {attempt} failed with error: {e}")

            # If we're not on the last attempt, wait and retry
            if attempt < max_retries:
                time.sleep(delay)
                delay *= 2  # Optional: exponential backoff

        # After all retries fail
        print("All attempts failed. Returning empty list.")
        return []

    def analyze_verification(self, prediction_query: str, all_sources: List[Dict]) -> Dict:
        """Analyze the sources to determine if the prediction was accurate."""
        article_summaries = "\n".join(
            [f"Title: {src['title']}, Source: {src['source']}, Description: {src['description']}" for src in all_sources]
        )

        print("Okay analyze_verification")
        system_prompt = """
        You are an AI analyst verifying predictions for Polymarket, a prediction market where users bet on real-world outcomes. Your task is to classify claims as TRUE, FALSE, or UNCERTAIN *only when evidence is insufficient*.

        ### Rules:
        1. *Classification Criteria*:
        - ⁠ TRUE ⁠: The news articles *conclusively confirm* the prediction happened (e.g., "Bill passed" → voting records show it passed).
        - ⁠ FALSE ⁠: The news articles *conclusively disprove* the prediction (e.g., "Company will move HQ" → CEO denies it).
        - ⁠ UNCERTAIN ⁠: *Only if* evidence is missing, conflicting, or outdated (e.g., no articles after the predicted event date).

        2. *Evidence Standards*:
        - Prioritize *recent articles* (within 7 days of prediction date).
        - Trust *primary sources* (government releases, official statements) over opinion pieces.
        - Ignore irrelevant or off-topic articles.

        3. *Conflict Handling*:
        - If sources conflict, weigh authoritative sources (e.g., Reuters) higher than fringe outlets.
        - If timing is unclear (e.g., "will happen next week" but no update), default to ⁠ UNCERTAIN ⁠.
        
        """       

        analysis_prompt = f"""
        The prediction is: "{prediction_query}". 

        Here are some recent news articles about this topic:
        {article_summaries}

        Based on this data, determine if the prediction was accurate. 
        Summarize the key evidence and provide the output in *JSON format* with the following structure:

        {{
          "result": "TRUE/FALSE/UNCERTAIN",
          "summary": "Brief explanation of why the claim is classified as TRUE, FALSE, or UNCERTAIN based on the news articles."
        }}

        Ensure the response is *valid JSON* with no additional text.
        """
        
        ai_verification = self.groq_client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": analysis_prompt},
            ],
        )
        
        match = re.search(r"\{(.*)\}", ai_verification.choices[0].message.content, re.DOTALL)
        if match:
            print("Match found")
            ai_verification_result = "{" + match.group(1) + "}"
            try:
                return json.loads(ai_verification_result)
            except json.JSONDecodeError:
                return {
                    "result": "UNCERTAIN",
                    "summary": "Could not analyze the prediction due to formatting issues."
                }
        else:
            return {
                "result": "UNCERTAIN",
                "summary": "Could not analyze the prediction due to formatting issues."
            }
    
    def verify_prediction(self, prediction_query: str) -> Dict:
        """Main method to verify a prediction."""
        # Generate search query
        search_query = self.generate_search_query(prediction_query)
        # search_query = prediction_query
        print(f"Generated Search Query: {search_query}")
        
        # Fetch news articles
        articles = self.fetch_news_articles(search_query)
        # print("articles", articles)
        # Fetch Google search results
        google_results = self.fetch_google_results(search_query)
        
        # Prepare sources from both APIs
        all_sources = [
            {"title": a['title'], "source": a['link'], "description": a['snippet']} for a in articles
        ] + [
            {"title": g['title'], "source": g['link'], "description": g['snippet']} for g in google_results
        ]

        # all_sources = [
        #     {"title": g['title'], "source": g['link'], "snippet": g['snippet']} for g in google_results
        # ]
        # print("all_sources", all_sources)
        if not all_sources:
            return {
                "result": "UNCERTAIN",
                "summary": "No relevant information found to verify this prediction.",
                "sources": []
            }
        # print("articles", articles)
        print("all_sources", len(all_sources))
        # Analyze verification
        verification_data = self.analyze_verification(prediction_query, all_sources)
        print("Final result")
        # Final result
        final_result = {
            "result": verification_data["result"],
            "summary": verification_data["summary"],
            "sources": all_sources
        }
        
        return final_result
    
# ============ COMPONENT 2: PREDICTOR PROFILE BUILDER ============

class PredictorProfiler:
    def __init__(self, groq_client, datura_api_key, datura_api_url):
        self.groq_client = groq_client
        self.datura_api_key = datura_api_key
        self.datura_api_url = datura_api_url

    async def build_user_profile(self, handle: str, max_retries: int = 5) -> Dict:
        print(handle)

        if handle.startswith("@"):
            handle = handle[1:]

        """Fetch recent tweets from a specific user."""
        headers = {
            "Authorization": f"{self.datura_api_key}",
            "Content-Type": "application/json",
        }
        
        params = {
            "user": handle,
            "query": "until:2024-9-28",
            "count": 100  #100
        }
        
        for attempt in range(max_retries):
            try:
                response = await asyncio.to_thread(requests.get, "https://apis.datura.ai/twitter/post/user", params=params, headers=headers)
                response.raise_for_status()
                tweets_ls = response.json()
                print(len(tweets_ls), "tweets found")
                if tweets_ls:
                    tweets = [tweet.get("text", "") for tweet in tweets_ls]
                    return {"tweets": tweets}
                
            except requests.exceptions.RequestException as e:
                return {"error": f"Failed to fetch tweets: {str(e)}", "tweets": []}
            
            print(f"Attempt {attempt + 1} failed. Retrying...")
            await asyncio.sleep(2)
        
        return {"error": "Invalid Username. No tweets found after 5 attempts.", "tweets": []}
    
    # async def filter_predictions(self, tweets: List[str]) -> Dict:
    #     """Filter tweets to only include predictions."""
    #     # tweets = tweets[:30]  # Limit to 30 tweets for analysis

    #     tweet_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(tweets)])
        
    #     system_context = """You are an expert in identifying explicit and implicit predictions in tweets that could be relevant to Polymarket, a prediction market platform. Polymarket users bet on future events in politics, policy, business, law, and geopolitics.

    #     **Definitions:**
    #     1. **Explicit Prediction**: A direct statement about a future outcome (e.g., 'X will happen,' 'Y is likely to pass').
    #     2. **Implicit Prediction**: A statement implying a future outcome (e.g., 'Senator proposes bill,' 'Protests may lead to...').

    #     **Polymarket Topics Include:**
    #     - Elections, legislation, court rulings
    #     - Policy changes (tariffs, regulations)
    #     - Business decisions (company moves, market impacts)
    #     - Geopolitical events (wars, treaties, sanctions)
    #     - Legal/Investigative outcomes (prosecutions, declassifications)

    #     **Exclude:**
    #     - Past events (unless they imply future consequences)
    #     - Pure opinions without forecastable outcomes
    #     - Non-actionable statements (e.g., 'People are struggling')

    #     **Examples:**
    #     - 'Trump will win in 2024' → **Yes (Explicit)**
    #     - 'Senator proposes bill to ban TikTok' → **Yes (Implicit)**
    #     - 'The economy is collapsing' → **No (No actionable prediction)**

    #     **Task:** For each tweet, return **'Yes'** if it contains an explicit/implicit prediction relevant to Polymarket, else **'No'**. Respond *only* with a JSON object like:
    #     {
    #     "predictions": ["Yes", "No", ...]
    #     }
    #     """
        
    #     response = await asyncio.to_thread(self.groq_client.chat.completions.create,
    #         model=MODEL_NAME,
    #         messages=[{"role": "system", "content": system_context},
    #                   {"role": "user", "content": tweet_list}]
    #     )
        
    #     raw_output = response.choices[0].message.content
        
    #     # Remove markdown wrapping if present
    #     if raw_output.startswith("\njson"):
    #         raw_output = re.sub(r"\njson|\n", "", raw_output).strip()
        
    #     try:
    #         parsed = json.loads(raw_output)
    #         return {
    #             "predictions": parsed.get("predictions", []),
    #         }
    #     except Exception as e:
    #         print("Failed to parse LLM response:")
    #         print(raw_output)
    #         raise e

    async def filter_predictions(self, tweets: List[str]) -> Dict:
        """Filter tweets to only include predictions, processing in batches of 25."""
        # Initialize an empty list to store all prediction results
        all_predictions = []
        batch_size = 25
        
        # Process tweets in batches of 25
        for i in range(0, len(tweets), batch_size):
            batch_tweets = tweets[i:i+batch_size]
            batch_tweet_list = "\n".join([f"{j+1}. {t}" for j, t in enumerate(batch_tweets)])
            
            system_context = """You are an expert in identifying explicit and implicit predictions in tweets that could be relevant to Polymarket, a prediction market platform. Polymarket users bet on future events in politics, policy, business, law, and geopolitics.

            **Definitions:**
            1. **Explicit Prediction**: A direct statement about a future outcome (e.g., 'X will happen,' 'Y is likely to pass').
            2. **Implicit Prediction**: A statement implying a future outcome (e.g., 'Senator proposes bill,' 'Protests may lead to...').

            **Polymarket Topics Include:**
            - Elections, legislation, court rulings
            - Policy changes (tariffs, regulations)
            - Business decisions (company moves, market impacts)
            - Geopolitical events (wars, treaties, sanctions)
            - Legal/Investigative outcomes (prosecutions, declassifications)

            **Exclude:**
            - Past events (unless they imply future consequences)
            - Pure opinions without forecastable outcomes
            - Non-actionable statements (e.g., 'People are struggling')

            **Examples:**
            - 'Trump will win in 2024' → **Yes (Explicit)**
            - 'Senator proposes bill to ban TikTok' → **Yes (Implicit)**
            - "Nikki Haley is gaining ground in Iowa polls." → Yes (Implicit) (implies prediction market relevance)
            - "Senate to vote on crypto regulation bill next week." → Yes (Implicit)
            - "Will Russia use nuclear weapons in 2024?" → Yes (Explicit)
            - "Israel expected to launch ground invasion of Gaza." → Yes (Implicit)
            - "Elon Musk hints at stepping down as Twitter CEO." → Yes (Implicit)
            - 'The economy is collapsing' → **No (No actionable prediction)**
            - "I miss when politicians actually cared about the people." → No (opinion, not predictive)
            - "The economy crashed last year and it's all downhill from here." → No (past event, vague future implication)
            - "Climate change is real." → No (statement, no actionable prediction)

            **Task:** For each tweet, return **'Yes'** if it contains an explicit/implicit prediction relevant to Polymarket. Respond *only* with a JSON object like:
            {
            "predictions": ["Yes", "No", ...]
            }
            """
            
            response = await asyncio.to_thread(self.groq_client.chat.completions.create,
                model=MODEL_NAME1,
                messages=[{"role": "system", "content": system_context},
                        {"role": "user", "content": batch_tweet_list}]
            )
            
            raw_output = response.choices[0].message.content
            print("raw_output", raw_output)
            raw_output = re.sub(r"^```(json)?|```$", "", raw_output).strip()
            # Step 2: Extract JSON Content (if extra text exists)
            match = re.search(r"\{.*\}", raw_output, re.DOTALL)
            if match:
                raw_output = match.group(0)  # Extract only the JSON content
            
            try:
                parsed = json.loads(raw_output.encode().decode('utf-8-sig'))  # Removes BOM if present
                # Extend the all_predictions list with the batch results
                all_predictions.extend(parsed.get("predictions", []))
            except json.JSONDecodeError as e:
                print(f"Failed to parse LLM response for batch {i//batch_size + 1}:")
                print(raw_output)
                # If parsing fails, add "No" for each tweet in the batch as a fallback
                all_predictions.extend(["No"] * len(batch_tweets))
        
        # Return combined results in the expected format
        return {
            "predictions": all_predictions,
        }

    # async def filter_predictions(self, tweets: List[str]) -> Dict:
    #     """Filter tweets to only include predictions, processing in batches of 25 with a boosting second check."""
    #     # Initialize an empty list to store all prediction results
    #     all_predictions = []
    #     batch_size = 25
        
    #     # Process tweets in batches of 25
    #     for i in range(0, len(tweets), batch_size):
    #         batch_tweets = tweets[i:i+batch_size]
    #         batch_tweet_list = "\n".join([f"{j+1}. {t}" for j, t in enumerate(batch_tweets)])
            
    #         system_context = """You are an expert in identifying explicit and implicit predictions in tweets that could be relevant to Polymarket, a prediction market platform. Polymarket users bet on future events in politics, policy, business, law, and geopolitics.

    #         **Definitions:**
    #         1. **Explicit Prediction**: A direct statement about a future outcome (e.g., 'X will happen,' 'Y is likely to pass').
    #         2. **Implicit Prediction**: A statement implying a future outcome (e.g., 'Senator proposes bill,' 'Protests may lead to...').

    #         **Polymarket Topics Include:**
    #         - Elections, legislation, court rulings
    #         - Policy changes (tariffs, regulations)
    #         - Business decisions (company moves, market impacts)
    #         - Geopolitical events (wars, treaties, sanctions)
    #         - Legal/Investigative outcomes (prosecutions, declassifications)

    #         **Exclude:**
    #         - Past events (unless they imply future consequences)
    #         - Pure opinions without forecastable outcomes
    #         - Non-actionable statements (e.g., 'People are struggling')

    #         **Examples:**
    #         - 'Trump will win in 2024' → **Yes (Explicit)**
    #         - 'Senator proposes bill to ban TikTok' → **Yes (Implicit)**
    #         - 'The economy is collapsing' → **No (No actionable prediction)**

    #         **Task:** For each tweet, return **'Yes'** if it contains an explicit/implicit prediction relevant to Polymarket, else **'No'**. Respond *only* with a JSON object like:
    #         {
    #         "predictions": ["Yes", "No", ...]
    #         }
    #         """
            
    #         try:
    #             response = await asyncio.to_thread(self.groq_client.chat.completions.create,
    #                 model=MODEL_NAME,
    #                 messages=[{"role": "system", "content": system_context},
    #                         {"role": "user", "content": batch_tweet_list}]
    #             )
                
    #             raw_output = response.choices[0].message.content
                
    #             print("raw_output", raw_output)
    #             raw_output = re.sub(r"^```(json)?|```$", "", raw_output).strip()
    #             # Step 2: Extract JSON Content (if extra text exists)
    #             match = re.search(r"\{.*\}", raw_output, re.DOTALL)
    #             if match:
    #                 raw_output = match.group(0)  # Extract only the JSON content
                
    #             parsed = json.loads(raw_output.encode().decode('utf-8-sig'))
    #             batch_predictions = parsed.get("predictions", [])
                
    #             # Second check (boosting) for tweets initially classified as "No"
    #             no_indices = [j for j, pred in enumerate(batch_predictions) if pred == "No"]
    #             if no_indices:
    #                 no_tweets = [batch_tweets[j] for j in no_indices]
                    
    #                 # Only proceed if there are "No" predictions to recheck
    #                 if no_tweets:
    #                     no_tweet_list = "\n".join([f"{j+1}. {t}" for j, t in enumerate(no_tweets)])
                        
    #                     # More permissive context for the second check to catch subtle predictions
    #                     boost_context = """You are reviewing tweets that were initially classified as NOT containing predictions. 
    #                     Be more generous and look very carefully for ANY hint of prediction about future events or implicit forecasts.
    #                     Even if the prediction is subtle or indirect, classify it as "Yes" if there's any reasonable interpretation as a prediction.
                        
    #                     **Examples of subtle predictions that should be "Yes":**
    #                     - "Things don't look good for the bill" (implies prediction about bill's failure)
    #                     - "Watch this space regarding the election" (implies upcoming election prediction)
    #                     - "The market doesn't seem to be pricing this in yet" (implies future market movement)
                        
    #                     For each tweet, respond **'Yes'** if there's ANY possible prediction interpretation, else **'No'**.
    #                     Respond *only* with a JSON object like:
    #                     {
    #                     "predictions": ["Yes", "No", ...]
    #                     }
    #                     """
                        
    #                     boost_response = await asyncio.to_thread(self.groq_client.chat.completions.create,
    #                         model=MODEL_NAME,
    #                         messages=[{"role": "system", "content": boost_context},
    #                                 {"role": "user", "content": no_tweet_list}]
    #                     )
                        
    #                     boost_output = boost_response.choices[0].message.content
    #                     print("boost_output", boost_output)
    #                     match = re.search(r"\{.*\}", boost_output, re.DOTALL)
    #                     if match:
    #                         boost_output = match.group(0)  # Extract only the JSON content
                        
    #                     boost_parsed = json.loads(boost_output.encode().decode('utf-8-sig'))                       
  
    #                     boost_predictions = boost_parsed.get("predictions", [])
                        
    #                     # Replace original "No" predictions with boosted results
    #                     for k, idx in enumerate(no_indices):
    #                         if k < len(boost_predictions) and boost_predictions[k] == "Yes":
    #                             batch_predictions[idx] = "Yes"
    #                             print(f"Boosted tweet to 'Yes': {batch_tweets[idx]}")
                
    #             # Add the batch predictions to the combined results
    #             all_predictions.extend(batch_predictions)
                
    #         except Exception as e:
    #             print(f"Failed to process batch {i//batch_size + 1}: {str(e)}")
    #             # If processing fails, add "No" for each tweet in the batch as a fallback
    #             all_predictions.extend(["No"] * len(batch_tweets))
        
    #     # Return combined results in the expected format
    #     return {
    #         "predictions": all_predictions,
    #     }
    
    async def apply_filter(self, tweets: List[str], outcomes: Dict) -> List[str]:
        """Apply prediction filter to tweets."""
        outcomes_list = outcomes["predictions"]
        zipped = list(zip(tweets, outcomes_list))
        filtered_tweets = [tweet for tweet, outcome in zipped if outcome == "Yes"]
        print(f"Filtered {len(filtered_tweets)} prediction tweets from {len(tweets)} total tweets.")
        return filtered_tweets
    
    async def analyze_prediction_patterns(self, filtered_tweets: List[str]) -> Dict:
        """Analyze patterns in the user's predictions."""
        if not filtered_tweets:
            return {
                "total_predictions": 0,
                "topics": {},
                "confidence_level": "N/A",
                "prediction_style": "N/A",
                "summary": "No predictions found for this user."
            }
        
        tweet_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(filtered_tweets)])
        
        analysis_prompt = f"""
        You are an expert analyst of prediction patterns and behaviors.  
        Analyze the following list of prediction tweets from a single user and provide a comprehensive analysis with the following information:

        1. The main topics this person makes predictions about (politics, crypto, sports, etc.)
        2. Their typical confidence level (certain, hedging, speculative)
        3. Their prediction style (quantitative, qualitative, conditional)
        4. Any patterns you notice in their prediction behavior

        Format your response as JSON:
        {{
            "topics": {{"topic1": percentage, "topic2": percentage, ...}},
            "confidence_level": "description of their confidence level",
            "prediction_style": "description of their prediction style",
            "patterns": ["pattern1", "pattern2", ...],
            "summary": "A brief summary of this predictor's profile"
        }}

        Ensure the response is **valid JSON** with no additional text.
        """
        
        response = await asyncio.to_thread(self.groq_client.chat.completions.create,
            model=MODEL_NAME,
            messages=[{"role": "system", "content": analysis_prompt},
                      {"role": "user", "content": tweet_list}]
        )
        
        raw_output = response.choices[0].message.content
        
        # Extract JSON from response
        match = re.search(r"\{(.*)\}", raw_output, re.DOTALL)
        json_content = "{" + match.group(1) + "}"
        
        try:
            analysis = json.loads(json_content)
            analysis["total_predictions"] = len(filtered_tweets)
            return analysis
        except json.JSONDecodeError:
            return {
                "total_predictions": len(filtered_tweets),
                "error": "Could not parse analysis",
                "raw_output": raw_output
            }
    
    async def build_profile(self, handle: str) -> Dict:
        """Main method to build a predictor's profile."""
        # Get user tweets
        user_data = await self.build_user_profile(handle)
        
        if "error" in user_data:
            return {"error": user_data["error"]}
        
        # Filter predictions
        prediction_outcomes = await self.filter_predictions(user_data["tweets"])
        
        # Apply filter
        filtered_predictions = await self.apply_filter(user_data["tweets"], prediction_outcomes)
        print("Filtered predictions build profile:", len(filtered_predictions))
        
        # Analyze prediction patterns
        analysis = await self.analyze_prediction_patterns(filtered_predictions)
        
        # Build complete profile
        profile = {
            "handle": handle,
            "total_tweets_analyzed": len(user_data["tweets"]),
            "prediction_tweets": filtered_predictions,
            "prediction_count": len(filtered_predictions),
            "prediction_rate": len(filtered_predictions) / len(user_data["tweets"]) if user_data["tweets"] else 0,
            "analysis": analysis
        }
        
        return profile

    async def build_profiles(self, handles: List[str]) -> List[Dict]:
        """Build profiles for multiple handles concurrently."""
        tasks = [self.build_profile(handle) for handle in handles]
        profiles = await asyncio.gather(*tasks)
        return profiles

    async def calculate_credibility_score(self, handle: str, prediction_verifier: PredictionVerifier) -> Dict:
        """Calculate credibility score asynchronously for a single handle."""
        # Await the profile retrieval
        profile = await self.build_profile(handle)

        if "error" in profile:
            print("This sucks")
            print("Error in profile:", profile["error"])
            return {"error": profile["error"]}

        if not profile["prediction_tweets"]:
            return {
                "handle": handle,
                "credibility_score": 0.0,
                "prediction_stats": {
                    "total": 0,
                    "true": 0,
                    "false": 0,
                    "uncertain": 0
                },
                "message": "No predictions found for this user."
            }

        # Track verification results
        verification_stats = {
            "total": len(profile["prediction_tweets"]),
            "true": 0,
            "false": 0,
            "uncertain": 0,
            "verifications": []
        }

        async def verify_prediction_async(prediction):
            """Run prediction verification in a separate thread (avoids blocking)."""
            return await asyncio.to_thread(prediction_verifier.verify_prediction, prediction)

        # Run all verifications concurrently
        verification_results = await asyncio.gather(
            *(verify_prediction_async(prediction) for prediction in profile["prediction_tweets"])
        )

        # Process verification results
        for prediction, verification in zip(profile["prediction_tweets"], verification_results):
            if verification["result"] == "TRUE":
                verification_stats["true"] += 1
            elif verification["result"] == "FALSE":
                verification_stats["false"] += 1
            else:  # UNCERTAIN
                verification_stats["uncertain"] += 1

            verification_stats["verifications"].append({
                "prediction": prediction,
                "result": verification["result"],
                "summary": verification["summary"],
                "sources": verification["sources"]
            })

        # Calculate credibility score
        if verification_stats["total"] > 0 and (verification_stats["true"] + verification_stats["false"]) > 0:
            credibility_score = verification_stats["true"] / (verification_stats["true"] + verification_stats["false"])
        else:
            credibility_score = 0.0

        # Create the final result
        result = {
            "handle": handle,
            "credibility_score": round(credibility_score, 2),
            "prediction_stats": {
                "total": verification_stats["total"],
                "true": verification_stats["true"],
                "false": verification_stats["false"],
                "uncertain": verification_stats["uncertain"]
            },
            "verified_predictions": verification_stats["verifications"],
            "profile_summary": profile["analysis"].get("summary", "")
        }

        return result

    async def calculate_credibility_scores_batch(self, handles: List[str], prediction_verifier: PredictionVerifier) -> List[Dict]:
        """Calculate credibility scores for multiple users concurrently."""
        tasks = [self.calculate_credibility_score(handle, prediction_verifier) for handle in handles]
        return await asyncio.gather(*tasks)


# ============ AUTOGEN INTEGRATION ============
# Register the functions with the agents

def find_predictions_wrapper(user_prompt: str):
    """Wrapper for the find_predictions function"""
    print("Finding predictions...")
    return asyncio.run(prediction_finder.find_predictions(user_prompt))
    
def build_profiles_wrapper(handles: List[str]):
    """Wrapper for the build_profiles function"""
    print("Building profiles...")
    return asyncio.run(predictor_profiler.build_profiles(handles))

def  calculate_credibility_scores_batch_wrapper(handles: List[str]):
    print("Calculating credibility scores for batch...")
    """Wrapper for the calculate_credibility_scores_batch function"""
    return asyncio.run(predictor_profiler.calculate_credibility_scores_batch(handles, prediction_verifier))

def verify_prediction_wrapper(prediction: str):
    print("Verifying prediction...")    
    """Wrapper for the verify_prediction function"""
    return prediction_verifier.verify_prediction(prediction)

function_definitions = [
    {
        "name": "find_predictions",
        "description": "Finds predictions based on a user prompt.",
        "parameters": {
            "type": "object",
            "properties": {
                "user_prompt": {
                    "type": "string",
                    "description": "The prompt provided by the user to find predictions."
                }
            },
            "required": ["user_prompt"]
        }
    },
    {
        "name": "build_profiles",
        "description": "Builds profiles for a list of handles.",
        "parameters": {
            "type": "object",
            "properties": {
                "handles": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "The list of handles (e.g., usernames or identifiers) for whom to build profiles."
                }
            },
            "required": ["handles"]
        }
    },
    {
        "name": "calculate_credibility_scores_batch",
        "description": "Calculates credibility scores for a batch of handles.",
        "parameters": {
            "type": "object",
            "properties": {
                "handles": {
                    "type": "array",
                    "items": {
                        "type": "string"
                    },
                    "description": "The list of handles (e.g., usernames or identifiers) for whom to calculate credibility scores."
                }
            },
            "required": ["handles"]
        }
    },
    {
        "name": "verify_prediction",
        "description": "Verifies a given prediction.",
        "parameters": {
            "type": "object",
            "properties": {
                "prediction": {
                    "type": "string",
                    "description": "The prediction text to be verified."
                }
            },
            "required": ["prediction"]
        }
    }
]

tools_schema = [func_def for func_def in function_definitions]

# Global variable to store the assistant agent
persistent_assistant = None

# Define Autogen agents
def create_prediction_agents():

    global persistent_assistant
    if persistent_assistant is not None:
        return persistent_assistant  # Reuse existing agent    
    # Initialize components
    prediction_finder = PredictionFinder(client, DATURA_API_KEY, DATURA_API_URL)
    predictor_profiler = PredictorProfiler(client, DATURA_API_KEY, DATURA_API_URL)
    prediction_verifier = PredictionVerifier(client, NEWS_API_TOKEN, GOOGLE_API_KEY, GOOGLE_CSE_ID)
    
    # Create function map for the UserProxyAgent with the new function
    function_map = {
        "find_predictions": find_predictions_wrapper,
        # "build_profile": build_profile_wrapper,
        "verify_prediction": verify_prediction_wrapper,
        # "calculate_credibility": calculate_credibility_wrapper,
        "build_profiles": build_profiles_wrapper,
        "calculate_credibility_scores_batch": calculate_credibility_scores_batch_wrapper
    }
    
    # Create the assistant agent with strict instructions
    assistant = AssistantAgent(
        name="SwarmcentsHelper",
        # llm_config=llm_config,
        model_client=client1,
        tools=[find_predictions_wrapper, build_profiles_wrapper, verify_prediction_wrapper, calculate_credibility_scores_batch_wrapper],
        system_message="""You are a prediction analysis expert that helps users find, profile, and verify predictions.
        
STRICT RULES YOU MUST FOLLOW:
1. You MUST ONLY use the provided functions - never make up data or predictions
2. You MUST ask clarifying questions if the user request is unclear. In that case , do not execute any function.
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
    
    persistent_assistant = assistant  # Store the assistant agent in the global variable
    return assistant
    

# Update the initiate_chat message too
async def run_prediction_analysis(query: str = "Give me a summary of the predictions made by @elonmusk"):
    # Create agents
    assistant = create_prediction_agents()

    response = await assistant.on_messages(
        [TextMessage(content=query, source="user")],
        cancellation_token=CancellationToken(),
    )
    print("inner", type(response.inner_messages))
    print("chat", type(response.chat_message))
    return response.chat_message.content

if __name__ == "__main__":
    # Example 1: Find predictions on a topic
    prediction_finder = PredictionFinder(client, DATURA_API_KEY, DATURA_API_URL)
    
    # Example 2: Build a predictor profile
    predictor_profiler = PredictorProfiler(client, DATURA_API_KEY, DATURA_API_URL)
    
    # Example 3: Verify a prediction
    prediction_verifier = PredictionVerifier(client, NEWS_API_TOKEN, GOOGLE_API_KEY, GOOGLE_CSE_ID)

    print("User: Hello")
    print()
    # response = asyncio.run(run_prediction_analysis("Given me predictions on Will trump lower tariffs on china in april?"))
    response = asyncio.run(run_prediction_analysis("Build profile history for @Polymarket"))

    print("Response from prediction analysis:")
    print(response)
    
    #print("User: You are looking awesome today")
    #print() @shaneyyricch
    # response = asyncio.run(run_prediction_analysis("Give me credibility scores for @elonmusk"))

    # print("Response from prediction analysis:")
    # print(response)
    # asyncio.run(run_prediction_analysis("Now give me credibility scores of the 1st 2 handles in a tabular format"))