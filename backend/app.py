from flask import Flask, request, jsonify
from flask_cors import CORS
import logging
from datetime import datetime
from typing import Dict, Any
from models import ModelHandler, ModelError
from config import Config
from utils import (
    setup_logging,
    validate_image,
    format_response,
    handle_error,
    validate_request_data,
    format_model_response
)
from functools import wraps
import time
from collections import defaultdict

# Configure logging
Config.setup_logging()
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": Config.CORS_ORIGINS}})

# Validate configuration
Config.validate_config()

# Initialize model handler
model_handler = ModelHandler({
    "OPENAI_API_KEY": Config.OPENAI_API_KEY,
    "GOOGLE_API_KEY": Config.GOOGLE_API_KEY,
    "GROQ_API_KEY": Config.GROQ_API_KEY
})

# Rate limiting
request_counts = defaultdict(list)

def rate_limit(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        now = time.time()
        client_ip = request.remote_addr
        
        # Clean old requests
        request_counts[client_ip] = [t for t in request_counts[client_ip] if now - t < 60]
        
        # Check rate limit
        if len(request_counts[client_ip]) >= Config.API_RATE_LIMIT:
            return jsonify(handle_error("Rate limit exceeded", 429)), 429
        
        # Add current request
        request_counts[client_ip].append(now)
        
        return f(*args, **kwargs)
    return decorated_function

@app.route('/api/analyze', methods=['POST'])
@rate_limit
def analyze_plant():
    """Analyze plant image and symptoms."""
    try:
        # Get and validate request data
        data = request.json
        validate_request_data(data)
        
        image_data = data.get('image')
        symptoms = data.get('symptoms', '')
        model_choice = data.get('model', 'GPT-4 Vision')
        region = data.get('region', 'Unknown')
        crop_type = data.get('crop_type', 'Unknown')

        # Determine query type
        query_type = "combined"
        if not image_data and symptoms:
            query_type = "text_only"
        elif image_data and not symptoms:
            query_type = "image_only"

        # Process and validate image if provided
        processed_image = None
        if image_data:
            try:
                processed_image = model_handler.process_image(image_data)
            except ValueError as e:
                return jsonify(handle_error(e, 400)), 400

        # Analyze based on model choice and query type
        try:
            if model_choice == "GPT-4 Vision":
                if query_type == "text_only":
                    result = model_handler.analyze_with_gpt4_text(symptoms, region, crop_type)
                elif query_type == "image_only":
                    result = model_handler.analyze_with_gpt4_image(processed_image, region, crop_type)
                else:
                    result = model_handler.analyze_with_gpt4(processed_image, symptoms, region, crop_type)
            elif model_choice == "Gemini":
                if query_type == "text_only":
                    result = model_handler.analyze_with_gemini_text(symptoms, region, crop_type)
                elif query_type == "image_only":
                    result = model_handler.analyze_with_gemini_image(processed_image, region, crop_type)
                else:
                    result = model_handler.analyze_with_gemini(processed_image, symptoms, region, crop_type)
            elif model_choice == "Groq":
                if query_type == "text_only":
                    result = model_handler.analyze_with_groq(symptoms, region, crop_type)
                else:
                    raise ValueError("Groq model only supports text-based analysis. Please provide a text description.")
            else:
                raise ValueError(f"Unsupported model choice: {model_choice}. Supported models are: GPT-4 Vision, Gemini, Groq")

            # Format and validate response
            formatted_result = format_model_response(result)
            
            # Add request metadata
            formatted_result["request_metadata"] = {
                "timestamp": datetime.utcnow().isoformat(),
                "model_used": model_choice,
                "region": region,
                "crop_type": crop_type,
                "model_version": Config.get_model_config()[model_choice.lower().replace("-", "").replace(" ", "")]["model"],
                "has_symptoms": bool(symptoms and symptoms.strip()),
                "has_image": bool(image_data),
                "query_type": query_type,
                "image_quality": formatted_result["plant_info"].get("image_quality", "N/A"),
                "is_plant_image": formatted_result["plant_info"].get("is_plant_image", True)
            }

            # Add user guidance based on the analysis
            if query_type == "image_only":
                formatted_result["user_guidance"]["additional_info_needed"] = "Please provide any symptoms or observations you've noticed about the plant."
            elif query_type == "text_only":
                formatted_result["user_guidance"]["image_improvement"] = "Consider uploading an image for more accurate analysis."
            else:
                if not formatted_result["plant_info"]["is_plant_image"]:
                    formatted_result["user_guidance"]["image_improvement"] = "Please upload an image of a plant or leaf for better analysis."
                elif formatted_result["plant_info"]["image_quality"] != "Good":
                    formatted_result["user_guidance"]["image_improvement"] = "Please try to take a clearer image of the plant or leaf."
                if not symptoms or not symptoms.strip():
                    formatted_result["user_guidance"]["additional_info_needed"] = "Please provide any symptoms or observations you've noticed about the plant."

            return jsonify(format_response(formatted_result))

        except ModelError as e:
            logger.error(f"Model analysis error: {str(e)}")
            return jsonify(handle_error(e, 500)), 500
        except Exception as e:
            logger.error(f"Analysis error: {str(e)}")
            return jsonify(handle_error(e, 500)), 500

    except Exception as e:
        logger.error(f"Request processing error: {str(e)}")
        return jsonify(handle_error(e, 500)), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify(format_response({
        'status': 'healthy',
        'version': '1.0.0',
        'timestamp': datetime.utcnow().isoformat(),
        'models': {
            'gpt4': Config.GPT4_MODEL,
            'gemini': Config.GEMINI_MODEL,
            'groq': Config.GROQ_MODEL
        },
        'capabilities': {
            'image_analysis': True,
            'text_analysis': True,
            'disease_detection': True,
            'user_guidance': True
        }
    }))

@app.route('/api/models', methods=['GET'])
def list_models():
    """List available AI models and their capabilities."""
    model_config = Config.get_model_config()
    return jsonify(format_response({
        'models': [
            {
                'name': 'GPT-4 Vision',
                'capabilities': ['image_analysis', 'text_analysis', 'disease_detection', 'user_guidance'],
                'max_image_size': f"{Config.MAX_IMAGE_SIZE/1024/1024}MB",
                'supported_formats': Config.SUPPORTED_FORMATS,
                'model_version': model_config['gpt4']['model'],
                'fallback_model': model_config['gpt4']['fallback_model'],
                'timeout': model_config['gpt4']['timeout'],
                'retry_attempts': model_config['gpt4']['retry_attempts'],
                'features': [
                    'High-quality image analysis',
                    'Detailed disease detection',
                    'Comprehensive treatment recommendations',
                    'User guidance and feedback'
                ]
            },
            {
                'name': 'Gemini',
                'capabilities': ['image_analysis', 'text_analysis', 'disease_detection', 'user_guidance'],
                'max_image_size': f"{Config.MAX_IMAGE_SIZE/1024/1024}MB",
                'supported_formats': Config.SUPPORTED_FORMATS,
                'model_version': model_config['gemini']['model'],
                'fallback_model': model_config['gemini']['fallback_model'],
                'timeout': model_config['gemini']['timeout'],
                'retry_attempts': model_config['gemini']['retry_attempts'],
                'features': [
                    'Fast image analysis',
                    'Accurate disease detection',
                    'Practical treatment solutions',
                    'User-friendly guidance'
                ]
            },
            {
                'name': 'Groq',
                'capabilities': ['text_analysis', 'disease_detection', 'user_guidance'],
                'max_image_size': 'N/A',
                'supported_formats': [],
                'model_version': model_config['groq']['model'],
                'fallback_model': model_config['groq']['fallback_model'],
                'timeout': model_config['groq']['timeout'],
                'retry_attempts': model_config['groq']['retry_attempts'],
                'features': [
                    'Text-based analysis',
                    'Quick disease detection',
                    'Basic treatment recommendations',
                    'Simple user guidance'
                ]
            }
        ]
    }))

if __name__ == '__main__':
    app.run(
        host=Config.HOST,
        port=Config.PORT,
        debug=Config.DEBUG
    ) 