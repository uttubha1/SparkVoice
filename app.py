from flask import Flask, request, jsonify
from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline, RobertaTokenizer
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Load the sentiment analysis pipeline with the desired model
model_name = "UtkarshFlairminds/sentimentAnalysis"
tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
model = AutoModelForSequenceClassification.from_pretrained(model_name)
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

        # Use the sentiment analysis pipeline
        sentiment_result = sentiment_analysis(transcription)

        result = {
            'transcript': transcription,
            'sentiment': sentiment_result,
        }

        return jsonify({'result': result})

if __name__ == '__main__':
    app.run(debug=True)
