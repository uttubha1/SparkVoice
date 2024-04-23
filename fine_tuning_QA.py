import torch
from transformers import XLMRobertaTokenizer, XLMRobertaForQuestionAnswering, AdamW, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
from tqdm import tqdm

# Check if CUDA (GPU support) is available
if torch.cuda.is_available():
    print("CUDA (GPU) is available!")
else:
    print("CUDA (GPU) is not available. Falling back to CPU.")

# Load your CoQA dataset from local files
train_data = pd.read_parquet(r"C:\Users\Admin\Downloads\train.parquet")
valid_data = pd.read_parquet(r"C:\Users\Admin\Downloads\validation.parquet")

# Load tokenizer and model
tokenizer = XLMRobertaTokenizer.from_pretrained("xlm-roberta-base")
model = XLMRobertaForQuestionAnswering.from_pretrained("xlm-roberta-base")

# Define batch size
batch_size = 8

# Preprocess data function
import json

def preprocess_data(data):
    input_texts = []
    start_positions = []
    end_positions = []
    
    for _, row in data.iterrows():
        story = row['story']
        questions = row['questions']
        answers = row['answers']

        for question, answer in zip(questions, answers['input_text']):
            input_text = story + " [SEP] " + question
            input_texts.append(input_text)
        
        start_positions.extend(answers['answer_start'])
        end_positions.extend(answers['answer_end'])

    input_ids = tokenizer(input_texts, padding=True, truncation=True, return_tensors="pt")
    start_positions = torch.tensor(start_positions, dtype=torch.long)
    end_positions = torch.tensor(end_positions, dtype=torch.long)
    
    return input_ids.input_ids, start_positions, end_positions


# Preprocess train and validation data
train_input_ids, train_start_positions, train_end_positions = preprocess_data(train_data)
valid_input_ids, valid_start_positions, valid_end_positions = preprocess_data(valid_data)

# Create TensorDatasets
train_dataset = TensorDataset(train_input_ids, train_start_positions, train_end_positions)
valid_dataset = TensorDataset(valid_input_ids, valid_start_positions, valid_end_positions)

# Define DataLoader
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=batch_size)

# Set device to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Move model to device
model.to(device)
print("Model's device:", next(model.parameters()).device)

# Define optimizer and scheduler
optimizer = AdamW(model.parameters(), lr=2e-5)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=len(train_loader))

# Training loop
epochs = 3
for epoch in range(epochs):
    model.train()
    for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        batch = tuple(t.to(device) for t in batch)
        input_ids, start_positions, end_positions = batch
        
        outputs = model(input_ids=input_ids, start_positions=start_positions, end_positions=end_positions)
        loss = outputs.loss
        
        loss.backward()
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

# Save fine-tuned model
output_dir = "fine_tuned_xlm_roberta_coqa"
model.save_pretrained(output_dir)
tokenizer.save_pretrained(output_dir)
print('Model saved at:', output_dir)
