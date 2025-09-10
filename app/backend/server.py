"""
HTTP Server for Stroke to LaTeX Conversion

This server accepts stroke data via POST requests and returns LaTeX expressions.
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import numpy as np
import time
import logging
import sys
sys.path.append('../..')
from data_tools.stroke_processing.stroke2img import strokes_to_image,save_as_bmp
from data_tools.stroke_processing.scale_strokes import rescale_strokes
from inference_single import Inference

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Global inference model (loaded once at startup)
inference_model = None
STROKE_LENGTH = 50  # pixels

def initialize_model():
    """Initialize the inference model at server startup."""
    global inference_model
    logger.info("Loading inference model...")
    start_time = time.time()
    inference_model = Inference()
    load_time = time.time() - start_time
    logger.info(f"Model loaded in {load_time:.2f} seconds")

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'model_loaded': inference_model is not None
    })

@app.route('/convert', methods=['POST'])
def convert_strokes_to_latex():
    """
    Convert strokes to LaTeX.
    
    Expected JSON payload:
    {
        "strokes": [
            [[x1, y1], [x2, y2], ...],  # First stroke
            [[x3, y3], [x4, y4], ...],  # Second stroke
            ...
        ],
        "stroke_length": 50  # Optional, defaults to 50
    }
    
    Returns:
    {
        "latex": "converted_expression",
        "processing_time": 0.123,
        "image_size": [width, height]
    }
    """
    try:
        # Parse request data
        data = request.get_json()
        if not data or 'strokes' not in data:
            return jsonify({'error': 'No strokes data provided'}), 400
        
        strokes = data['strokes']
        stroke_length = data.get('stroke_length', STROKE_LENGTH)
        
        if not strokes:
            return jsonify({'error': 'Empty strokes array'}), 400
        
        # Start timing
        start_time = time.time()
        
        # Rescale strokes
        strokes_norm, size = rescale_strokes(strokes, stroke_length)
        
        # Convert to image
        img = strokes_to_image(
            strokes_norm, 
            image_size=(int(size[0])+4, int(size[1])+4), 
            line_thickness=2, 
            padding=2
        )
        
        # save_as_bmp(img, 'debug.bmp')  # For debugging

        # Convert to LaTeX
        latex = inference_model.convert2latex(img)
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Return result
        return jsonify({
            'latex': latex,
            'processing_time': processing_time,
            'image_size': [int(size[0])+4, int(size[1])+4]
        })
        
    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    import sys
    import os
    
    # Get port from environment variable, command line, or default
    port = int(os.environ.get('PORT', sys.argv[1] if len(sys.argv) > 1 else 5001))
    
    # Initialize model before starting server
    initialize_model()
    
    # Check if running in production (App Runner/Docker)
    is_production = os.environ.get('FLASK_ENV') == 'production'
    
    # Start Flask server
    logger.info(f"Starting server on http://0.0.0.0:{port} (Production: {is_production})")
    
    if is_production:
        # In production, use gunicorn (installed in Docker)
        # This will be handled by the Docker CMD, so Flask serves as fallback
        app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
    else:
        # Development mode
        app.run(host='0.0.0.0', port=port, debug=False)