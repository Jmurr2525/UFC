import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2Tokenizer, GPT2LMHeadModel
from max.engine import InferenceSession, Model
from memory_profiler import profile
# Set up the path to your JSON files
data_folder = '../output'

def load_data(directory, num_files):
    fight_data = []
    file_count = 0
    
    # Get the list of JSON files in the directory
    json_files = [file for file in os.listdir(directory) if file.endswith('.json')]
    
    # Sort the JSON files alphabetically
    json_files.sort()
    
    # Iterate over the first num_files JSON files
    for file_name in json_files[:num_files]:
        file_path = os.path.join(directory, file_name)
        try:
            with open(file_path, 'r') as file:
                data = json.load(file)
                fight_data.extend(data)
                file_count += 1
        except Exception as e:
            print(f"An error occurred while loading file {file_path}: {e}")
    
    print(f"Loaded {file_count} files.")
    
    return fight_data


# Create a custom dataset
class FightDataset(Dataset):
    def __init__(self, fight_data, tokenizer, max_length):
        self.fight_data = fight_data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.fight_data)

    def __getitem__(self, idx):
        fight = self.fight_data[idx]
        fighter_names = ' vs '.join(fight['fighter_names'])
        round_descriptions = ' '.join(fight['round_descriptions'])
        input_text = f"{fighter_names}\n{round_descriptions}"

        encoding = self.tokenizer(input_text, truncation=True, max_length=self.max_length, padding='max_length', return_tensors='pt')
        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()

        return input_ids, attention_mask

# Train the model
def train_model(model, dataloader, optimizer, epochs, checkpoint_path, accumulation_steps=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    for epoch in range(epochs):
        for batch_idx, batch in enumerate(dataloader):
            input_ids, attention_mask = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

            outputs = model(input_ids, attention_mask=attention_mask, labels=input_ids)
            loss = outputs.loss
            loss = loss / accumulation_steps
            loss.backward()

            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if (batch_idx + 1) % (100 * accumulation_steps) == 0:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': loss.item()
                }, checkpoint_path)

            torch.cuda.empty_cache()

        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

    model.save_pretrained('fight_model_normal')
# Generate fight descriptions
def generate_description(model, tokenizer, max_length, num_rounds):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    fighter_names = input("Enter fighter names (Fighter 1 vs Fighter 2): ")
    seed_text = f"{fighter_names}\n"

    input_ids = tokenizer.encode(seed_text, return_tensors='pt').to(device)

    with torch.no_grad():
        output = model.generate(input_ids, max_length=max_length, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=tokenizer.eos_token_id)

    generated_description = tokenizer.decode(output[0], skip_special_tokens=True)

    # Split the generated description into rounds
    rounds = generated_description.split('\n')
    rounds = rounds[:num_rounds+1]  # Include fighter names and specified number of rounds
    generated_description = '\n'.join(rounds)

    return generated_description
# Main function
def main():
    # Load and preprocess the data
    fight_data = load_data(data_folder, num_files=100)

    # Load the pre-trained GPT-2 tokenizer and model
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    model = GPT2LMHeadModel.from_pretrained('gpt2')

    # Set the padding token
    tokenizer.pad_token = tokenizer.eos_token

    # Create a custom dataset and data loader
    max_length = 1024
    dataset = FightDataset(fight_data, tokenizer, max_length)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

    # Set up the optimizer
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    # Check if checkpoint exists
    checkpoint_path = 'checkpoint.pth'
    if os.path.exists(checkpoint_path):
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        loss = checkpoint['loss']
        print(f"Resuming training from epoch {start_epoch}, loss: {loss:.4f}")
    else:
        start_epoch = 0

    # Train the model
    epochs = 10
    accumulation_steps = 4
    train_model(model, dataloader, optimizer, epochs, checkpoint_path, accumulation_steps)

    # Generate new fight descriptions
    num_rounds = 3
    generated_description = generate_description(model, tokenizer, max_length, num_rounds)

    # Print the generated fight description
    print("\nGenerated Fight Description:")
    print(generated_description)

if __name__ == '__main__':
    main()