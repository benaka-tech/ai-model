from flask import Flask, request, jsonify, g
from flask_cors import CORS
from models import load_and_quantize_models, load_ocr_reader
import logging
from dotenv import load_dotenv
import torch
from PIL import Image
import io
import json
import time
import psutil
import os
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
try:
    import numpy as np
    logger.info("Numpy is available.")
except ImportError as e:
    logger.error("Numpy is not available: %s", str(e))

load_dotenv()
app = Flask(__name__)
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@app.before_request
def before_request():
    g.start_time = time.time()

def log_request_metrics(response):
    latency = time.time() - g.start_time
    app.logger.info(f"Endpoint: {request.path}, Method: {request.method}, Latency: {latency:.2f}s, Status: {response.status_code}")
    return response

@app.route('/metrics', methods=['GET'])
def get_metrics():
    latency = time.time() - g.start_time if hasattr(g, 'start_time') else None
    throughput = 1 / latency if latency else None
    error_rate = 0
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    memory_usage = memory_info.percent
    available_memory = memory_info.available / (1024 ** 2)
    total_memory = memory_info.total / (1024 ** 2)

    disk_usage = psutil.disk_usage('/')
    disk_usage_percent = disk_usage.percent

    network_io = psutil.net_io_counters()
    bytes_sent = network_io.bytes_sent
    bytes_received = network_io.bytes_recv

    metrics = {
        'latency': latency,
        'throughput': throughput,
        'error_rate': error_rate,
        'cpu_usage': cpu_usage,
        'memory_usage': memory_usage,
        'available_memory': available_memory,
        'total_memory': total_memory,
        'disk_usage_percent': disk_usage_percent,
        'bytes_sent': bytes_sent,
        'bytes_received': bytes_received
    }
    return jsonify(metrics)

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
            
        answer = qa_model.answer_question(context=data['context'], question=data['question'])
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
            
        sentiment, confidence = sentiment_model.analyze_sentiment(data['text'])
        return jsonify({'sentiment': sentiment, 'confidence': confidence})
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
        predictions = [imagenet_classes[idx] for idx in indices[0].tolist()]
        return jsonify({'predictions': predictions})
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

# Fine-tune a sentiment analysis model
@app.route('/fine-tune-sentiment', methods=['POST'])
def fine_tune_sentiment():
    try:
        # Load the public synthetic-student-feedback dataset
        dataset = load_dataset("janerkegb/synthetic-student-feedback")
        tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

        # Map the `intent` column to `labels` and convert to integers
        label_mapping = {
            "POSITIVE_FEEDBACK": 0,
            "NEGATIVE_FEEDBACK": 1,
            "TEACHING_QUALITY": 2,
            "EXAM_DIFFICULTY": 3,
            "HOMEWORK_LOAD": 4,
            "LECTURE_SPEED": 5,
            "NEUTRAL_COMMENT": 6,
            "COURSE_MATERIALS": 7,
            "SUGGESTION": 8
        }
        dataset = dataset.map(lambda x: {"labels": label_mapping[x["intent"]]})

        # Split the dataset into train and test sets if 'test' split is not available
        if "test" not in dataset:
            dataset = dataset["train"].train_test_split(test_size=0.2, seed=42)

        # Adjust the column name based on the dataset structure
        def preprocess_function(examples):
            return tokenizer(examples["feedback_text"], truncation=True, padding='max_length', max_length=128)

        tokenized_datasets = dataset.map(preprocess_function, batched=True)

        # Load pre-trained model
        model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=len(label_mapping))

        # Define training arguments
        training_args = TrainingArguments(
            output_dir="./results",
            evaluation_strategy="epoch",
            learning_rate=2e-5,
            per_device_train_batch_size=4,
            num_train_epochs=1,
            weight_decay=0.01,
            save_strategy="epoch",
            logging_dir="./logs",
            logging_steps=10,
        )

        # Initialize Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_datasets["train"].shuffle(seed=42),
            eval_dataset=tokenized_datasets["test"].shuffle(seed=42)
        )

        # Train the model
        trainer.train()

        # Save the fine-tuned model
        model.save_pretrained("./fine_tuned_sentiment_model")
        tokenizer.save_pretrained("./fine_tuned_sentiment_model")

        return jsonify({"message": "Model fine-tuned and saved successfully."})
    except Exception as e:
        logger.error(f"Fine-tuning error: {str(e)}")
        return jsonify({"error": str(e)}), 500

# Load fine-tuned model and tokenizer
try:
    fine_tuned_model = DistilBertForSequenceClassification.from_pretrained("./fine_tuned_sentiment_model")
    fine_tuned_tokenizer = DistilBertTokenizer.from_pretrained("./fine_tuned_sentiment_model")
except Exception as e:
    logger.error(f"Error loading fine-tuned model: {str(e)}")
    fine_tuned_model = None
    fine_tuned_tokenizer = None

# Inference function
def predict_sentiment(text):
    if fine_tuned_model is None or fine_tuned_tokenizer is None:
        return "Model not loaded"

    inputs = fine_tuned_tokenizer(text, padding=True, truncation=True, max_length=73, return_tensors="pt")
    with torch.no_grad():
        outputs = fine_tuned_model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    sentiment_label = torch.argmax(predictions).item()
    return "Positive" if sentiment_label == 1 else "Negative"

# API endpoint for sentiment prediction
@app.route('/predict-sentiment', methods=['POST'])
def predict_sentiment_route():
    try:
        data = request.get_json()
        text = data['text']
        sentiment = predict_sentiment(text)
        return jsonify({"sentiment": sentiment})
    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)