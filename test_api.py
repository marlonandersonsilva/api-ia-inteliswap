import requests
import json
import sys
import platform

def test_predictions():
    """Test if the prediction endpoints are working correctly"""
    print("Testing prediction endpoints...")
    
    # Test health endpoint
    try:
        response = requests.get("http://localhost:5000/health")
        if response.status_code != 200:
            print(f"Health check failed: {response.status_code}")
            return False
            
        health_status = response.json()
        print(f"Health check: {health_status}")
        
        # Check if models are available
        models_available = health_status.get('modelos', {})
        
        # Test Random Forest endpoint if available
        if models_available.get('random_forest'):
            # Replace with appropriate test features for your model
            test_data = {"features": [1, 2, 3, 4]}
            
            response = requests.post(
                "http://localhost:5000/predict_rf",
                json=test_data
            )
            
            if response.status_code == 200:
                print(f"Random Forest prediction successful: {response.json()}")
            else:
                print(f"Random Forest prediction failed: {response.status_code}, {response.text}")
                return False
        else:
            print("Random Forest model not available")
        
        # Test XGBoost endpoint if available
        if models_available.get('xgboost'):
            # Replace with appropriate test features for your model
            test_data = {"features": [1, 2, 3, 4]}
            
            response = requests.post(
                "http://localhost:5000/predict_xgb",
                json=test_data
            )
            
            if response.status_code == 200:
                print(f"XGBoost prediction successful: {response.json()}")
            else:
                print(f"XGBoost prediction failed: {response.status_code}, {response.text}")
                return False
        else:
            print("XGBoost model not available")
            
        return True
                
    except Exception as e:
        print(f"Error testing API: {str(e)}")
        return False

if __name__ == "__main__":
    print("Running API tests...")
    success = test_predictions()
    if success:
        print("All tests passed!")
        sys.exit(0)
    else:
        print("Tests failed!")
        sys.exit(1)