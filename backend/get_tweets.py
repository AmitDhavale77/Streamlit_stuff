from datura_py import Datura

DATURA_API_KEY = "dt_$X6oACKtNOE_2RL984Dg-C8Ds6HZmsQLA4N7ez3NysVg"
# datura = Datura(api_key=DATURA_API_KEY)
# result = datura.basic_twitter_search(
#         query="",
#         sort="Top",
#         user="elonmusk",
#         lang="en",
#         verified=True,
#         blue_verified=False,
#         is_quote=False,
#         is_video=False,
#         is_image=False,
#         min_retweets=1,
#         min_replies=1,
#         min_likes=1,
#         count=10
#     )
import requests


url = "https://apis.datura.ai/twitter/post/user"
headers = {
    "Authorization": DATURA_API_KEY,
    "Content-Type": "application/json"
}
params = {
    "user": "PolymarketSpike",
    # "query": "until:2024-10-28",
    "count": 3
}

response = requests.get(url, params=params, headers=headers)

print(response.status_code)
print(response.json())
