import os
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
import logging

# Setup logging to see the model's decisions and potential errors
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def load_historical_data(student_dir):
    """Load historical test data from JSON files in a given directory."""
    try:
        historical_data = []
        if not os.path.exists(student_dir):
            logger.warning(f"Student directory '{student_dir}' not found. No historical data to load.")
            return []
            
        for file_name in os.listdir(student_dir):
            if file_name.startswith('results_') and file_name.endswith('.json'):
                file_path = os.path.join(student_dir, file_name)
                try:
                    with open(file_path, 'r') as f:
                        data = json.load(f)
                        # Ensure all required keys exist before appending
                        if all(k in data for k in ['accuracy', 'average_time']):
                            historical_data.append(data)
                except Exception as e:
                    logger.error(f"Error reading or parsing {file_path}: {e}")
        logger.debug(f"Loaded {len(historical_data)} historical results from {student_dir}")
        return historical_data
    except Exception as e:
        logger.error(f"Error loading historical data from {student_dir}: {e}")
        return []

def predict_pass_fail(accuracy, avg_time, student_dir):
    """Predict pass/fail based on accuracy and average time using historical data."""
    try:
        historical_data = load_historical_data(student_dir)
        
        # Use a default model if there's not enough historical data for training
        if len(historical_data) < 3:
            logger.warning("Insufficient historical data for prediction; using default model.")
            X_train = np.array([[0.5, 10], [0.7, 8], [0.3, 15]]) # Default data
            y_train = np.array([0, 1, 0])  # Default labels: 0=fail, 1=pass
        else:
            X_train = np.array([[d['accuracy'], d['average_time']] for d in historical_data])
            y_train = np.array([1 if d['accuracy'] >= 0.6 else 0 for d in historical_data])
        
        model = LogisticRegression()
        model.fit(X_train, y_train)
        
        # Predict on the new data point
        prediction = model.predict([[accuracy, avg_time]])
        logger.debug(f"Predict pass/fail: accuracy={accuracy}, avg_time={avg_time}, prediction={prediction[0]}")
        return 'Pass' if prediction[0] == 1 else 'Fail'
    except Exception as e:
        logger.error(f"Error in pass/fail prediction: {e}")
        return "Prediction Error"
