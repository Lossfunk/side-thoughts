import requests

def get_remaining_credits(client):
    url = "https://openrouter.ai/api/v1/auth/key"
    headers = {
        "Authorization": f"Bearer {client.api_key}"
    }
    response = requests.get(url, headers=headers)
    data = response.json()
    print(f"remaining credits: {data['data']['limit_remaining']}")
