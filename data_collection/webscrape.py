import requests
from bs4 import BeautifulSoup
import json

def scrape_fight_data(url):
    # Headers to mimic a browser visit
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3'
    }

    # Send a request to the website
    response = requests.get(url, headers=headers)

    # Check if the request was successful
    if response.status_code != 200:
        print("Failed to retrieve the webpage")
        return []

    # Parse the content with BeautifulSoup
    soup = BeautifulSoup(response.content, 'html.parser')

    # Find all the 'event' divs
    event_divs = soup.find_all('div', class_='event')

    if not event_divs:
        print("No event divs found")
        return []

    # List to store the extracted data for all events
    events_data = []

    # Iterate over each event div
    for event_div in event_divs:
        # Extract fighter names
        fighter_names = []
        fighter_links = event_div.find_all('a', href=lambda href: href and href.startswith('/fighter/'))
        for link in fighter_links:
            fighter_names.append(link.get_text(strip=True))

        # Extract round descriptions
        round_descriptions = []
        round_headers = event_div.find_all('h3')
        for header in round_headers:
            round_num = header.get_text(strip=True)
            round_desc = header.find_next_sibling(string=True).strip()
            round_descriptions.append(f"{round_num}: {round_desc}")

        # Extract the official result
        result_header = event_div.find('h3', string='The Official Result')
        if result_header:
            official_result = result_header.find_next_sibling(string=True).strip()
        else:
            official_result = "Official result not found"

        # Create a dictionary for the current event's data
        event_data = {
            'fighter_names': fighter_names,
            'round_descriptions': round_descriptions,
            'official_result': official_result
        }

        # Append the event's data to the list of all events if it contains valid data
        if fighter_names and round_descriptions:
            events_data.append(event_data)
        else:
            print(f"Skipping event due to missing data: {event_data}")

    return events_data

# URL of the webpage to scrape
url = 'https://www.sherdog.com/news/news/UFC-295-Prochazka-vs-Pereira-PlaybyPlay-Results-Round-Scoring-191686'

# Scrape the fight data from the webpage
fight_data = scrape_fight_data(url)

# Clean up the fight data by removing events with missing data
cleaned_fight_data = [event for event in fight_data if event['fighter_names'] and event['round_descriptions']]

# Save the cleaned fight data as a JSON file
with open('fight_data_cleaned.json', 'w') as file:
    json.dump(cleaned_fight_data, file, indent=4)