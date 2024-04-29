import requests
from bs4 import BeautifulSoup

# URL of the webpage to scrape
url = 'https://en.wikipedia.org/wiki/List_of_UFC_events'

# Headers to mimic a browser visit
headers = {'User-Agent': 'Mozilla/5.0'}

# Send a request to the website
response = requests.get(url, headers=headers)

# Parse the HTML content
soup = BeautifulSoup(response.content, 'html.parser')

# Find the specific elements containing the event titles
event_elements = soup.find_all('a', href=True, title=True)

# Extract titles and write them to a file
with open('output.txt', 'w', encoding='utf-8') as file:
    for event in event_elements:
        if 'UFC' in event.text:  # Checking if 'UFC' is in the text of the element
            file.write(event.text + '\n')

print('Event titles written to output.txt')
