import streamlit as st
import requests
import json
from PIL import Image
import io
import base64
import time

# API endpoint
API_URL = "http://127.0.0.1:8000/api"

# Set page config
st.set_page_config(
    page_title="Plant Disease Detection System",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #4CAF50;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #45a049;
    }
    .result-card {
        padding: 1.5rem;
        border-radius: 10px;
        background-color: #f8f9fa;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .info-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #e8f5e9;
        margin: 0.5rem 0;
    }
    .warning-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #fff3e0;
        margin: 0.5rem 0;
    }
    .error-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #ffebee;
        margin: 0.5rem 0;
    }
    .success-box {
        padding: 1rem;
        border-radius: 5px;
        background-color: #e8f5e9;
        margin: 0.5rem 0;
    }
    .header {
        text-align: center;
        margin-bottom: 2rem;
    }
    .header h1 {
        color: #2e7d32;
        font-size: 2.5rem;
        margin-bottom: 0.5rem;
    }
    .header p {
        color: #666;
        font-size: 1.1rem;
    }
    .model-info {
        padding: 1rem;
        border-radius: 5px;
        background-color: #f5f5f5;
        margin: 1rem 0;
    }
    .loading {
        text-align: center;
        padding: 2rem;
    }
    .loading-spinner {
        width: 50px;
        height: 50px;
        border: 5px solid #f3f3f3;
        border-top: 5px solid #4CAF50;
        border-radius: 50%;
        animation: spin 1s linear infinite;
    }
    @keyframes spin {
        0% { transform: rotate(0deg); }
        100% { transform: rotate(360deg); }
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.markdown("### About")
    st.markdown("""
        This system uses advanced AI to detect plant diseases and provide treatment recommendations.
        
        **Features:**
        - Image-based disease detection
        - Multiple AI models
        - Detailed analysis
        - Treatment recommendations
        - Prevention measures
    """)
    
    st.markdown("### How to Use")
    st.markdown("""
        1. Upload a clear image of the plant
        2. Describe any symptoms you've noticed
        3. Select an AI model
        4. Click Analyze
    """)
    
    st.markdown("### Tips")
    st.markdown("""
        - Take clear, well-lit photos
        - Include both healthy and affected areas
        - Describe symptoms in detail
        - Try different models for best results
    """)

# Header
st.markdown('<div class="header"><h1>🌿 Plant Disease Detection System</h1><p>Upload an image of your plant and describe any symptoms you\'ve noticed. Our AI will analyze the image and provide detailed information about the plant and any potential diseases.</p></div>', unsafe_allow_html=True)

# Initialize session state variables at the start of the app
if 'analysis_type' not in st.session_state:
    st.session_state.analysis_type = None

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    # Image upload
    st.markdown("### 📸 Upload Plant Image")
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of the plant or leaf you want to analyze"
    )
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Image size warning
        if uploaded_file.size > 5 * 1024 * 1024:  # 5MB
            st.warning("⚠️ Image size is large. For better results, please upload a smaller image.")

with col2:
    # Text input for symptoms
    st.markdown("### 📝 Symptoms Description")
    symptoms = st.text_area(
        "Describe the symptoms or observations you've noticed:",
        height=150,
        help="Please describe any changes, discoloration, spots, or other symptoms you've observed on the plant"
    )
    
    # Model selection
    st.markdown("### 🤖 AI Model Selection")
    model_choice = st.selectbox(
        "Select an AI model for analysis:",
        ["GPT-4 Vision", "Gemini", "Groq"],
        help="GPT-4 Vision: Best for detailed analysis\nGemini: Fast and accurate\nGroq: Text-based analysis only"
    )
    
    # Model info
    with st.expander("Model Information"):
        if model_choice == "GPT-4 Vision":
            st.markdown("""
                **GPT-4 Vision**
                - High-quality image analysis
                - Detailed disease detection
                - Comprehensive treatment recommendations
                - Best for complex cases
            """)
        elif model_choice == "Gemini":
            st.markdown("""
                **Gemini**
                - Fast image analysis
                - Accurate disease detection
                - Practical treatment solutions
                - Good for quick results
            """)
        else:
            st.markdown("""
                **Groq**
                - Text-based analysis
                - Quick disease detection
                - Basic treatment recommendations
                - Good for simple cases
            """)

# Analysis button
if st.button("🔍 Analyze Plant", type="primary"):
    if uploaded_file is None and not symptoms:
        st.error("⚠️ Please provide either an image or a description of the plant/symptoms.")
        st.info("💡 You can:")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
                **Upload an image to:**
                - Identify the plant
                - Detect diseases
                - Get visual analysis
            """)
        with col2:
            st.markdown("""
                **Or describe:**
                - Plant symptoms
                - Your observations
                - Questions about plants
            """)
    else:
        # Determine analysis type based on available inputs
        if uploaded_file is None:
            st.session_state.analysis_type = "text_only"
            st.warning("⚠️ No image provided. Analysis will be based on text description only.")
            st.info("💡 For better results, consider uploading an image of the plant.")
        elif not symptoms:
            st.session_state.analysis_type = "image_only"
            st.warning("⚠️ No symptoms or observations provided. Analysis will be based on image only.")
            st.info("💡 For better results, please describe any symptoms or observations you've noticed.")
        else:
            st.session_state.analysis_type = "combined"

        # Proceed with analysis
        with st.spinner("Analyzing..."):
            try:
                # Convert image to base64 if available
                img_str = None
                if uploaded_file:
                    buffered = io.BytesIO()
                    image.save(buffered, format="PNG")
                    img_str = base64.b64encode(buffered.getvalue()).decode()

                # Prepare request data
                data = {
                    "image": f"data:image/png;base64,{img_str}" if img_str else None,
                    "symptoms": symptoms,
                    "model": model_choice
                }

                # Make API request
                response = requests.post(f"{API_URL}/analyze", json=data)
                
                if response.status_code == 200:
                    result = response.json()
                    data = result.get("data", {})

                    # Display results
                    st.markdown("### 📊 Analysis Results")

                    # Query Type and Intent
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown("#### 🔍 Analysis Type")
                    query_type = data.get("query_type", "combined").replace("_", " ").title()
                    query_intent = data.get("query_intent", "Disease detection").replace("_", " ").title()
                    response_focus = data.get("response_focus", "Plant analysis").replace("_", " ").title()

                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.markdown(f"**Query Type:** {query_type}")
                    with col2:
                        st.markdown(f"**Intent:** {query_intent}")
                    with col3:
                        st.markdown(f"**Focus:** {response_focus}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Plant Information
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown("#### 🌿 Plant Information")
                    plant_info = data.get("plant_info", {})
                    
                    # Check if image is of a plant
                    if not plant_info.get("is_plant_image", True):
                        st.error("⚠️ The uploaded image does not appear to be of a plant or leaf. Please upload a clear image of the plant.")
                    else:
                        # Display plant info in a grid
                        col1, col2 = st.columns(2)
                        with col1:
                            st.markdown(f"**Species:** {plant_info.get('species', 'Unknown')}")
                            st.markdown(f"**Common Name:** {plant_info.get('common_name', 'Unknown')}")
                            st.markdown(f"**Health Status:** {plant_info.get('health_status', 'Unknown')}")
                        with col2:
                            st.markdown(f"**Confidence:** {plant_info.get('confidence', 0) * 100:.1f}%")
                            st.markdown(f"**Image Quality:** {plant_info.get('image_quality', 'Unknown')}")
                        
                            # Image quality feedback
                            if plant_info.get("image_quality") != "Good":
                                st.warning(f"⚠️ {plant_info.get('image_feedback', '')}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Disease Information
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown("#### 🦠 Disease Information")
                    disease_info = data.get("disease_info", {})
                    
                    # Display disease info in a grid
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Disease Name:** {disease_info.get('name', 'No disease detected')}")
                        st.markdown(f"**Severity:** {disease_info.get('severity', 'Unknown')}")
                        st.markdown(f"**Analysis Type:** {disease_info.get('analysis_type', 'Unknown')}")
                    with col2:
                        st.markdown(f"**Confidence:** {disease_info.get('confidence', 0) * 100:.1f}%")

                    # Display symptoms
                    if disease_info.get("symptoms"):
                        st.markdown("**Observed Symptoms:**")
                        for symptom in disease_info["symptoms"]:
                            st.markdown(f"- {symptom}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Treatment
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown("#### 🧪 Treatment")
                    treatment = data.get("treatment", {})
                    
                    # Display treatment methods
                    if treatment.get("methods"):
                        st.markdown("**Recommended Methods:**")
                        for method in treatment["methods"]:
                            st.markdown(f"- {method}")

                    # Display products
                    if treatment.get("products"):
                        st.markdown("**Recommended Products:**")
                        for product in treatment["products"]:
                            st.markdown(f"- {product}")

                    # Display instructions
                    if treatment.get("instructions"):
                        st.markdown("**Application Instructions:**")
                        st.markdown(treatment["instructions"])
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Prevention
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown("#### 🛡️ Prevention")
                    prevention = data.get("prevention", {})
                    
                    # Display prevention measures
                    if prevention.get("measures"):
                        st.markdown("**Prevention Measures:**")
                        for measure in prevention["measures"]:
                            st.markdown(f"- {measure}")

                    # Display best practices
                    if prevention.get("best_practices"):
                        st.markdown("**Best Practices:**")
                        for practice in prevention["best_practices"]:
                            st.markdown(f"- {practice}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Additional Information
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown("#### ℹ️ Additional Information")
                    additional_info = data.get("additional_info", {})
                    
                    # Display notes
                    if additional_info.get("notes"):
                        st.markdown("**Notes:**")
                        st.markdown(additional_info["notes"])

                    # Display recommendations
                    if additional_info.get("recommendations"):
                        st.markdown("**Recommendations:**")
                        for rec in additional_info["recommendations"]:
                            st.markdown(f"- {rec}")

                    # Display next steps
                    if additional_info.get("next_steps"):
                        st.markdown("**Next Steps:**")
                        for step in additional_info["next_steps"]:
                            st.markdown(f"- {step}")

                    # Display references
                    if additional_info.get("references"):
                        st.markdown("**References:**")
                        for ref in additional_info["references"]:
                            st.markdown(f"- {ref}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # User Guidance
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown("#### 💡 Recommendations")
                    user_guidance = data.get("user_guidance", {})
                    
                    # Display image improvement suggestions
                    if user_guidance.get("image_improvement"):
                        st.markdown("**Image Improvement:**")
                        st.markdown(user_guidance["image_improvement"])

                    # Display additional info needed
                    if user_guidance.get("additional_info_needed"):
                        st.markdown("**Additional Information Needed:**")
                        st.markdown(user_guidance["additional_info_needed"])

                    # Display follow-up questions
                    if user_guidance.get("follow_up_questions"):
                        st.markdown("**Follow-up Questions:**")
                        for question in user_guidance["follow_up_questions"]:
                            st.markdown(f"- {question}")
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Metadata
                    st.markdown('<div class="result-card">', unsafe_allow_html=True)
                    st.markdown("#### 📝 Analysis Details")
                    metadata = data.get("request_metadata", {})
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown(f"**Model Used:** {metadata.get('model_used', 'Unknown')}")
                        st.markdown(f"**Region:** {metadata.get('region', 'Unknown')}")
                        st.markdown(f"**Crop Type:** {metadata.get('crop_type', 'Unknown')}")
                    with col2:
                        st.markdown(f"**Analysis Time:** {metadata.get('timestamp', 'Unknown')}")
                        st.markdown(f"**Model Version:** {metadata.get('model_version', 'Unknown')}")
                        st.markdown(f"**Confidence Score:** {metadata.get('confidence_score', 0) * 100:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)

                else:
                    error_data = response.json()
                    st.error(f"⚠️ Error: {error_data.get('error', {}).get('message', 'Unknown error occurred')}")
                    if 'suggestion' in error_data.get('error', {}):
                        st.info(f"💡 Suggestion: {error_data['error']['suggestion']}")

            except Exception as e:
                st.error(f"⚠️ An error occurred during analysis: {str(e)}")
                st.info("💡 Please try again with a different image or model.")
            finally:
                # Reset session state after analysis
                st.session_state.analysis_type = None

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center'>
        <p>Built with ❤️ using Streamlit, Flask, and AI</p>
        <p>For best results, please provide clear images and detailed symptom descriptions.</p>
    </div>
""", unsafe_allow_html=True) 