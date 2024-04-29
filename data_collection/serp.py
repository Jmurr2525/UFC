import os
from dotenv import load_dotenv
import requests

# Load environment variables from .env file
load_dotenv()

def get_search_results(query):
    # Your API endpoint
<<<<<<< HEAD:data_collection/serp.py
    url = 'https://api.spaceserp.com/google/search'

    # Access the API key
    api_key = os.getenv("SERP_API_KEY")

    # Check if API key is not found
    if not api_key:
        print("API key not found. Please check your .env file.")
        return None
=======
  
>>>>>>> 7c7325b14d19dca8e3148db2101b7613ddb5d248:serp.py

    # Query parameters
    params = {
        'q': query,
        'apiKey': api_key,
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
