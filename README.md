# Advanced Crop Disease Detection System

An AI-powered application that helps farmers and gardeners identify plant diseases and get treatment recommendations using computer vision and natural language processing. Built with a modern microservices architecture using Flask and Streamlit.

## Architecture

The application is built with a modern microservices architecture:

- **Frontend**: Streamlit-based web interface
- **Backend**: Flask REST API with LangChain integration
- **AI Models**: Support for multiple AI models (GPT-4 Vision, Gemini, Groq)

## Features

- Upload plant images for analysis
- Describe symptoms and observations
- Choose between multiple AI models (GPT-4 Vision, Gemini, Groq)
- Get detailed analysis including:
  - Plant identification with scientific names
  - Disease detection with confidence scores
  - Treatment recommendations (organic + chemical)
  - Prevention strategies
  - Regional and climate-specific advice
- Modern, intuitive user interface
- Advanced error handling and logging
- Region-aware analysis
- Confidence scoring system

## Prerequisites

- Python 3.8+
- OpenAI API key
- Google API key (for Gemini)
- Groq API key
- Git

## Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd crop-disease-detection
```

2. Create and activate a virtual environment:
```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the root directory:
```
OPENAI_API_KEY=your_openai_api_key
GOOGLE_API_KEY=your_google_api_key
GROQ_API_KEY=your_groq_api_key
```

## Running the Application

1. Start the backend server:
```bash
cd backend
python app.py
```

2. In a new terminal, start the frontend:
```bash
cd frontend
streamlit run app.py
```

3. Access the application at `http://localhost:8501`

## Testing the Application

### 1. Health Check
```bash
curl http://localhost:5000/api/health
```
Expected response:
```json
{
    "status": "healthy",
    "timestamp": "2024-03-14T12:00:00.000Z",
    "version": "1.0.0"
}
```

### 2. List Available Models
```bash
curl http://localhost:5000/api/models
```
Expected response:
```json
{
    "models": [
        {
            "name": "GPT-4 Vision",
            "capabilities": ["image_analysis", "text_analysis", "disease_detection"],
            "max_image_size": "20MB",
            "supported_formats": ["jpg", "jpeg", "png"]
        },
        ...
    ]
}
```

### 3. Test Plant Analysis
```bash
curl -X POST http://localhost:5000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "image": "base64_encoded_image",
    "symptoms": "Yellow spots and curling on tomato leaves",
    "model": "GPT-4 Vision",
    "region": "Gujarat",
    "crop_type": "Tomato"
  }'
```

Expected response structure:
```json
{
    "plant_info": {
        "common_name": "Tomato",
        "scientific_name": "Solanum lycopersicum",
        "family": "Solanaceae",
        "confidence": 0.95
    },
    "disease_info": {
        "is_diseased": true,
        "disease_name": "Early Blight",
        "pathogen_type": "Fungus",
        "confidence": 0.85,
        "symptoms": ["Yellow spots", "Leaf curling"],
        "severity": "Moderate"
    },
    "treatment": {
        "organic": [...],
        "chemical": [...]
    },
    "prevention": {
        "cultural_practices": [...],
        "monitoring": [...],
        "early_intervention": [...]
    },
    "additional_info": {
        "climate_considerations": [...],
        "regional_advice": [...],
        "economic_impact": "..."
    },
    "metadata": {
        "analysis_timestamp": "...",
        "model_version": "1.0",
        "confidence_score": 0.91
    }
}
```

## Example Usage

1. **Basic Analysis**:
   - Upload a plant image
   - Enter symptoms: "Yellow spots and curling on leaves"
   - Select GPT-4 Vision model
   - Click "Analyze Plant"

2. **Region-Specific Analysis**:
   - Add region: "Gujarat"
   - Add crop type: "Tomato"
   - Get region-specific recommendations

3. **Multiple Model Comparison**:
   - Try the same image with different models
   - Compare confidence scores and recommendations

## Troubleshooting

1. **Image Upload Issues**:
   - Ensure image is in supported format (jpg, jpeg, png)
   - Check image size (max 20MB for GPT-4 Vision)
   - Verify image integrity

2. **API Connection Issues**:
   - Check if backend server is running
   - Verify API keys in .env file
   - Check network connectivity

3. **Analysis Errors**:
   - Check error logs in backend
   - Verify input data format
   - Ensure sufficient API credits

## Development

### Project Structure
```
crop-disease-detection/
├── backend/
│   ├── app.py
│   ├── models.py
│   └── prompts.py
├── frontend/
│   └── app.py
├── requirements.txt
└── README.md
```

### Adding New Features
1. Create feature branch
2. Implement changes
3. Update tests
4. Submit pull request

## Contributing

1. Fork the repository
2. Create feature branch
3. Commit changes
4. Push to branch
5. Create pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details. 