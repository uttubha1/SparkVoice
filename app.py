from flask import Flask, request, jsonify
from transformers import RobertaTokenizer, RobertaForSequenceClassification, Trainer, TrainingArguments, pipeline
from datasets import load_dataset
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Fine-tune the RoBERTa model on the SST-2 dataset
dataset = load_dataset("sst")
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

def tokenize_function(examples):
    tokenized_inputs = tokenizer(examples["sentence"], padding="max_length", truncation=True)
    # Rename the 'label' column to 'labels' for Trainer compatibility
    tokenized_inputs["labels"] = [1 if label else 0 for label in examples["label"]]
    return tokenized_inputs

tokenized_datasets = dataset.map(tokenize_function, batched=True)

model = RobertaForSequenceClassification.from_pretrained("roberta-base", num_labels=2)
training_args = TrainingArguments("sst2", evaluation_strategy="epoch")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["validation"],
)
trainer.train()


# Load the fine-tuned model for sentiment analysis
sentiment_analysis = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)

# Configure the upload folder
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/analyze', methods=['POST'])
def analyze():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        with open(file_path, 'r', encoding='utf-8') as f:
            transcription = f.read()
        
        max_length = 512
        if len(transcription) > max_length:
            transcription = transcription[:max_length]

        # Use the fine-tuned model for sentiment analysis
        sentiment_result = sentiment_analysis(transcription)

        result = {
            'transcript': transcription,
            'sentiment': sentiment_result,
        }

        return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
