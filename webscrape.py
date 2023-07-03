import requests
from bs4 import BeautifulSoup

def scrape_website(url):
    # Define headers
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.3"
    }

    # Send a GET request to the website
    response = requests.get(url, headers=headers)

    # Open the output file in write mode
    with open('outputs/result.txt', 'w') as f:
        # If the GET request is successful, the status code will be 200
        if response.status_code == 200:
            # Get the content of the response
            webpage = response.text

            # Create a BeautifulSoup object and specify the parser
            soup = BeautifulSoup(webpage, "html.parser")

            # Find all the h3 tags
            h3_tags = soup.find_all("h3")

            # For each h3 tag
            for h3 in h3_tags:
                # Get the next sibling (the text following the h3 tag)
                next_text = h3.next_sibling

                # If next_text is not None, write the text to the file
                if next_text is not None:
                    f.write(next_text.strip() + '\n')
        else:
            f.write(f"Failed to retrieve webpage. Status code: {response.status_code}\n")