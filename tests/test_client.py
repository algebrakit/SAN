"""
Test client for the stroke to LaTeX conversion server.
"""

import requests
import json
from strokes.inkml_parser import InkMLParser

root = 'data/CHROME/CROHME_test_2011'
file1 = 'formulaire050-equation040'
file2 = 'formulaire050-equation070'
file3 = 'Inkdata_temp_InkFR_HPR_EQU_NOC_scc2_fi4_db135763'
file4 = 'Inkdata_temp_InkFR_HPR_EQU_NOC_scc54_fi4_db138003'
file5 = 'Inkdata_temp_InkFR_HPR_EQU_NOC_scc3_fi4_db135773'
file = f'{root}/{file3}.inkml'

def test_server(port=5001):
    """Test the server with sample data."""
    
    # Server URL
    base_url = f"http://localhost:{port}"
    
    # Check health
    response = requests.get(f"{base_url}/health")
    print("Health check:", response.json())
    
    # Load sample strokes from InkML file
    parser = InkMLParser()
    strokes, label = parser.parse_file(file)

    # Test with rescaling
    print("\n=== Testing with rescaling ===")
    payload = {
        "strokes": strokes,
        "stroke_length": 25
    }

    # log the payload in json
    print("Payload:", json.dumps(payload))    

    response = requests.post(
        f"{base_url}/convert",
        json=payload,
        headers={'Content-Type': 'application/json'}
    )
    
    if response.status_code == 200:
        result = response.json()
        print(f"LaTeX: {result['latex']}")
        print(f"Processing time: {result['processing_time']:.3f} seconds")
        print(f"Image size: {result['image_size']}")
    else:
        print(f"Error: {response.status_code}")
        print(response.text)
    


if __name__ == "__main__":
    test_server()