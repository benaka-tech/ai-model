from flask import Flask, request, jsonify
from flask_cors import CORS  # Add this import
from models import (
    load_and_quantize_models,
    load_ocr_reader
)
import logging
from dotenv import load_dotenv
import torch
from PIL import Image
import io
import numpy as np

load_dotenv()
app = Flask(__name__)
# Enable CORS with specific origins
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://expert-telegram-vr5jx9rxwgghxq77-8001.app.github.dev",
            "http://localhost:8001"
        ],
        "methods": ["OPTIONS", "GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load models at startup
summarizer, qa_model, sentiment_model, image_model, image_transform = load_and_quantize_models()
ocr_reader = load_ocr_reader()

@app.route('/summarize', methods=['POST'])
def summarize_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
            
        text = data['text']
        inputs = summarizer(text, max_length=130, min_length=30, return_tensors="pt")
        summary_ids = summarizer.generate(inputs['input_ids'])
        summary = summarizer.decode(summary_ids[0], skip_special_tokens=True)
        return jsonify({'summary': summary})
    except Exception as e:
        logger.error(f"Summarization error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/qa', methods=['POST'])
def answer_question():
    try:
        data = request.get_json()
        if not data or 'context' not in data or 'question' not in data:
            return jsonify({'error': 'Missing context or question parameter'}), 400
            
        inputs = qa_model.tokenizer(data['question'], data['context'], 
                            return_tensors='pt', 
                            truncation=True, 
                            max_length=512)
        with torch.no_grad():
            outputs = qa_model(**inputs)
        
        answer_start = torch.argmax(outputs.start_logits)
        answer_end = torch.argmax(outputs.end_logits) + 1
        answer = qa_model.tokenizer.convert_tokens_to_string(
            qa_model.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0][answer_start:answer_end])
        )
        return jsonify({'answer': answer})
    except Exception as e:
        logger.error(f"QA error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/sentiment', methods=['POST'])
def analyze_sentiment():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text parameter'}), 400
            
        inputs = sentiment_model.tokenizer(data['text'], 
                                   return_tensors='pt',
                                   truncation=True,
                                   max_length=512)
        with torch.no_grad():
            outputs = sentiment_model(**inputs)
        
        prediction = torch.argmax(outputs.logits).item()
        sentiment = 'positive' if prediction == 1 else 'negative'
        return jsonify({'sentiment': sentiment})
    except Exception as e:
        logger.error(f"Sentiment analysis error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/classify-image', methods=['POST'])
def classify_image():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file type'}), 400
            
        image = Image.open(io.BytesIO(file.read()))
        image = image_transform(image).unsqueeze(0)
        
        with torch.no_grad():
            outputs = image_model(image)
        
        # Get top 3 predictions
        _, indices = torch.topk(outputs, 3)
        return jsonify({'predictions': indices.tolist()[0]})
    except Exception as e:
        logger.error(f"Image classification error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/ocr', methods=['POST'])
def extract_text():
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
            
        file = request.files['file']
        if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            return jsonify({'error': 'Invalid file type'}), 400
            
        image = Image.open(io.BytesIO(file.read()))
        results = ocr_reader.readtext(np.array(image))
        text = ' '.join([result[1] for result in results])
        return jsonify({'text': text})
    except Exception as e:
        logger.error(f"OCR error: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)