# ai-model

## Overview
The `ai-model` project is designed to provide various AI-powered functionalities, including sentiment analysis, text summarization, question answering, image classification, and optical character recognition (OCR). It leverages state-of-the-art machine learning models and frameworks to deliver accurate and efficient results.

## Features
- **Sentiment Analysis**: Fine-tune and analyze sentiment from text data using pre-trained models like DistilBERT.
- **Text Summarization**: Generate concise summaries from long-form text.
- **Question Answering (QA)**: Answer questions based on provided context using NLP models.
- **Image Classification**: Classify images into predefined categories using deep learning models.
- **Optical Character Recognition (OCR)**: Extract text from images or scanned documents.

## Dataset
The project uses publicly available datasets for training and evaluation:
- **Synthetic Student Feedback Dataset**: A dataset containing synthetic feedback from students, categorized into multiple intents such as `POSITIVE_FEEDBACK`, `NEGATIVE_FEEDBACK`, `TEACHING_QUALITY`, etc.

## Technologies Used
- **Python 3**: The primary programming language for the project.
- **Hugging Face Transformers**: For pre-trained NLP models and fine-tuning.
- **PyTorch**: For deep learning model training and inference.
- **Flask**: A lightweight web framework for building RESTful APIs.
- **Datasets Library**: For loading and processing datasets.

## AI Models Used
- **DistilBERT**: A smaller, faster, and lighter version of BERT, used for tasks like sentiment analysis, text summarization, and question answering.
- **ResNet**: A deep residual network used for image classification tasks.
- **Tesseract OCR**: An open-source OCR engine used for extracting text from images.

### Benefits of Using These Models
1. **DistilBERT**:
   - Efficient and lightweight, making it suitable for real-time applications.
   - Pre-trained on large datasets, ensuring high accuracy for NLP tasks.
   - Fine-tunable for specific use cases like sentiment analysis and QA.

2. **ResNet**:
   - High performance in image classification tasks due to its deep architecture.
   - Handles vanishing gradient problems effectively, enabling better training of deep networks.


3. **Tesseract OCR**:
   - Accurate text extraction from images and scanned documents.
   - Supports multiple languages and is highly customizable..


   