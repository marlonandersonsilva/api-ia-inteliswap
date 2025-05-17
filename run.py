import os
import sys
import requests
import json
import time
import platform

def test_api():
    """Test if the prediction endpoints are working correctly"""
    print("Testing API endpoints...")
    
    # Wait for the server to start
    time.sleep(2)
    
    # Test health endpoint
    try:
        response = requests.get("http://localhost:5000/health")
        health_status = response.json()
        print(f"Health check: {health_status}")
        
        # Check if models are available
        models_available = health_status.get('modelos', {})
        
        # Test Random Forest endpoint if available
        if models_available.get('random_forest'):
            test_data = {"features": [1, 2, 3, 4]}  # Replace with appropriate test features
            response = requests.post(
                "http://localhost:5000/predict_rf",
                json=test_data
            )
            if response.status_code == 200:
                print(f"Random Forest prediction successful: {response.json()}")
            else:
                print(f"Random Forest prediction failed: {response.status_code}, {response.text}")
        
        # Test XGBoost endpoint if available
        if models_available.get('xgboost'):
            test_data = {"features": [1, 2, 3, 4]}  # Replace with appropriate test features
            response = requests.post(
                "http://localhost:5000/predict_xgb",
                json=test_data
            )
            if response.status_code == 200:
                print(f"XGBoost prediction successful: {response.json()}")
            else:
                print(f"XGBoost prediction failed: {response.status_code}, {response.text}")
                
    except Exception as e:
        print(f"Error testing API: {str(e)}")

def main():
    print("Starting ML Model API Backend...")
    
    # Use python3 on Linux and python on Windows
    if platform.system() == "Linux":
        os.system("python3 app.py")
    else:
        os.system("python app.py")

if __name__ == "__main__":
    main()