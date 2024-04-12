import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the API key
api_key = os.getenv("API_KEY")
import requests
import json

def get_search_results(query):
    # Your API endpoint
  

    # Your API key
    #Comment
    api_key = os.getenv("API_KEY")
    # Query parameters
    params = {
        'q': query,
        'key': api_key,
    }

    # Make the request
    response = requests.get(url, params=params)

# Check if request is successful
    if response.status_code == 200:
        json_response = response.json()
        if 'organic_results' in json_response and len(json_response['organic_results']) > 0:
            return json_response['organic_results'][0]['link']  # Return the link of the first result
        else:
            print('No results found.')
            return None
    else:
        print(f'Request failed with status {response.status_code}.')
        return None

#def main():
    query = 'sherdog sandhagen vs vera'  # The search term
    result = get_search_results(query)
    
    if result:
        # Print the link of the first result
        print(result)

#if __name__ == '__main__':
    main()
