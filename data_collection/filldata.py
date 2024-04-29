import os
import json
from data_collection.webscrape import scrape_fight_data
from data_collection.serp import get_search_results

def search(arg):
    result = get_search_results("sherdog play by play " + arg)
    return result

def main():
    input_file = 'input.txt'
    output_folder = 'output'

    # Create the output folder if it does not exist
    os.makedirs(output_folder, exist_ok=True)

    with open(input_file, 'r') as file:
        for line in file:
            line = line.strip()  # Remove trailing newline characters
            url = search(line)

            if url:
                fight_data = scrape_fight_data(url)

                if fight_data:
                    # Use the line from input.txt as the event name
                    event_name = line

                    # Sanitize the event name to make it a valid filename
                    filename = "".join([c for c in event_name if c.isalpha() or c.isdigit() or c == ' ']).rstrip() + '.json'

                    # Construct the full path for the output file
                    output_file_path = os.path.join(output_folder, filename)

                    # Write the fight data to the JSON file
                    with open(output_file_path, 'w') as output_file:
                        json.dump(fight_data, output_file, indent=4)
                else:
                    print(f"No data found for: {line}")
            else:
                print(f"No search results found for: {line}")

if __name__ == "__main__":
    main()