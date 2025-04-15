from typing import List, Dict
import asyncio 
import requests
from .PredictionVerifier import PredictionVerifier
import re
import json
import os 

# Initialize environment variables
MODEL_NAME = os.environ.get("MODEL_NAME", "gpt-4o-2024-08-06")

# ============ COMPONENT 2: PREDICTOR PROFILE BUILDER ============

class PredictionProfiler:
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
            # "sort": "Top",
            # "lang": "en",
            # "verified": False,
            # "blue_verified": False,
            # "is_quote": False,
            # "is_video": False,
            # "is_image": False,
            # "min_retweets": 0,
            # "min_replies": 0,
            # "min_likes": 0,
            "count": 50  #100
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

    async def filter_predictions(self, tweets: List[str]) -> Dict:
        """Filter tweets to only include predictions."""
        # tweets = tweets[:30]  # Limit to 30 tweets for analysis

        tweet_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(tweets)])
        
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
        - 'The economy is collapsing' → **No (No actionable prediction)**

        **Task:** For each tweet, return **'Yes'** if it contains an explicit/implicit prediction relevant to Polymarket, else **'No'**. Respond *only* with a JSON object like:
        {
        "predictions": ["Yes", "No", ...]
        }
        """
        
        response = await asyncio.to_thread(self.groq_client.chat.completions.create,
            model=MODEL_NAME,
            messages=[{"role": "system", "content": system_context},
                      {"role": "user", "content": tweet_list}]
        )

        raw_output = response.choices[0].message.content

        raw_output = re.sub(r"^```(json)?|```$", "", raw_output).strip()

        # Step 2: Extract JSON Content (if extra text exists)
        match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if match:
            raw_output = match.group(0)  # Extract only the JSON content

        try:
            # Step 3: Ensure valid encoding and parse JSON
            parsed = json.loads(raw_output.encode().decode('utf-8-sig'))  # Removes BOM if present
            return {"predictions": parsed.get("predictions", [])}

        except json.JSONDecodeError as e:
            print("JSON Decode Error:", str(e))
            print("Raw LLM Output:", repr(raw_output))  # Show raw output to debug
            raise e
        # raw_output = response.choices[0].message.content
        
        # # Remove markdown wrapping if present
        # if raw_output.startswith("\njson"):
        #     raw_output = re.sub(r"\njson|\n", "", raw_output).strip()
        
        # try:
        #     parsed = json.loads(raw_output)
        #     return {
        #         "predictions": parsed.get("predictions", []),
        #     }
        # except Exception as e:
        #     print("Failed to parse LLM response:")
        #     print(raw_output)
        #     raise e

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
        if verification_stats["total"] > 0:
            credibility_score = verification_stats["true"] / verification_stats["total"]
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

