from transformers import (
    BartForConditionalGeneration,
    BartTokenizer,
    DistilBertForQuestionAnswering,  # Updated import
    DistilBertTokenizer,             # Added for consistency
    DistilBertForSequenceClassification,
    DistilBertTokenizer
)
from torchvision.models import mobilenet_v2
from torchvision.transforms import Compose, Resize, ToTensor, Normalize
import easyocr
import torch

def load_summarization_model():
    """Load BART model for text summarization"""
    model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
    return model, tokenizer  # Return both model and tokenizer instead of lambda

def load_qa_model():
    """Load DistilBERT model for question answering"""
    model = DistilBertForQuestionAnswering.from_pretrained('distilbert-base-uncased-distilled-squad')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-distilled-squad')
    return model, tokenizer

def load_sentiment_model():
    """Load DistilBERT model for sentiment analysis"""
    model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased-finetuned-sst-2-english')
    return model, tokenizer

def load_image_model():
    """Load MobileNetV2 model for image classification"""
    model = mobilenet_v2(pretrained=True)
    model.eval()
    transform = Compose([
        Resize(224),  # MobileNet input size
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    return model, transform

def load_ocr_reader():
    """Load EasyOCR reader for text extraction"""
    return easyocr.Reader(['en'], gpu=False)

# Quantization function
def quantize_model(model):
    """Quantize the model to reduce size"""
    model.eval()
    return torch.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)

# Example of loading and quantizing models
def load_and_quantize_models():
    """Load and quantize all models"""
    # Load models and tokenizers
    summarization_model, summarization_tokenizer = load_summarization_model()
    qa_model, qa_tokenizer = load_qa_model()
    sentiment_model, sentiment_tokenizer = load_sentiment_model()
    image_model, image_transform = load_image_model()
    
    # Quantize models
    summarization_model = quantize_model(summarization_model)
    qa_model = quantize_model(qa_model)
    sentiment_model = quantize_model(sentiment_model)
    
    # Create a model wrapper that includes both model and tokenizer
    class ModelWrapper:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
        
        def __call__(self, *args, **kwargs):
            return self.tokenizer(*args, **kwargs)
        
        def generate(self, *args, **kwargs):
            return self.model.generate(*args, **kwargs)
        
        def decode(self, *args, **kwargs):
            return self.tokenizer.decode(*args, **kwargs)
    
    # Wrap models with their tokenizers
    summarizer = ModelWrapper(summarization_model, summarization_tokenizer)
    qa = ModelWrapper(qa_model, qa_tokenizer)
    sentiment = ModelWrapper(sentiment_model, sentiment_tokenizer)
    
    return summarizer, qa, sentiment, image_model, image_transform