from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd
import logging
from logging.handlers import RotatingFileHandler

# Initializing our application
app = Flask(__name__)

# Set up logging
log_handler = RotatingFileHandler('app.log', maxBytes=10000, backupCount=3)  # Rotate logs every 10KB, keep 3 backups
log_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
log_handler.setFormatter(formatter)
app.logger.addHandler(log_handler)

# Loading the trained models
model = joblib.load('models/model_lightgbm_log.pkl')
scaler = joblib.load('scalers/scaler_lightgbm_log.pkl')  

# List of feature names (you can adjust this based on your model's features)
feature_names = ['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'Latitude', 'Longitude']
features_to_scale = ['MedInc', 'AveRooms', 'AveBedrms', 'Population', 'AveOccup', 'OccupPerRoom', 'BedrmsPerRoom']

def feature_engineering(data):
    """ Perform feature engineering on input data. """
    filtered_data = {key: value for key, value in data.items() if key in feature_names}
    df = pd.DataFrame([filtered_data])
    df['OccupPerRoom'] = df['AveOccup'] / df['AveRooms']
    df['BedrmsPerRoom'] = df['AveBedrms'] / df['AveRooms']
    app.logger.debug(f"Filtered data for feature engineering: {filtered_data}")
    return df

def feature_scaling(df):
    """ Scale the necessary features and return the scaled dataframe. """
    df_features_to_scale = df[features_to_scale]
    app.logger.debug(f"Data to scale: {df_features_to_scale.head()}")
    
    # Scaling features
    scaled_features = scaler.transform(df_features_to_scale)
    scaled_df = pd.DataFrame(scaled_features, columns=features_to_scale, index=df_features_to_scale.index)

    # Concatenate scaled features with the remaining ones
    scaled_df = pd.concat([scaled_df, df.drop(columns=features_to_scale)], axis='columns')
    app.logger.debug(f"Scaled features: {scaled_df.head()}")
    
    return scaled_df

# Home route to serve the landing page (just information or welcome page)
@app.route('/')
def home():
    app.logger.info("Accessed home page.")
    return render_template('home.html')

# Prediction route to handle JSON text input for prediction
@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get the JSON data from the user input
            data = request.get_json()  # Parse the incoming JSON data
            app.logger.info(f"Received data for prediction: {data}")  # Debugging: Log the received data

            if not data:
                raise ValueError("No JSON data received")

            # Ensure the data contains the correct keys (feature names)
            missing_features = [feature for feature in feature_names if feature not in data]
            
            if missing_features:
                error_message = f"Missing the following feature(s): {', '.join(missing_features)}"
                print(error_message)
                return jsonify({"error": error_message}), 400

            # Ensure the data contains the correct keys (feature names)
            if all(feature in data for feature in feature_names): 
                # Perform feature engineering
                df = feature_engineering(data)

                # Scale the features
                scaled_df = feature_scaling(df)

                # Make prediction
                prediction = model.predict(scaled_df)
                app.logger.info(f"Prediction result: {prediction[0]}")

                # Return the predicted price (multiplied by 100,000 as per the model's output)
                return jsonify({"predicted_price": 100000 * prediction[0]})

            else:
                error_msg = "Missing or incorrect feature data."
                app.logger.error(error_msg)
                return jsonify({"error": error_msg}), 400

        except Exception as e:
            app.logger.exception("Error occurred during prediction.")
            return jsonify({"error": str(e)}), 400

    # If GET request, show the form
    return render_template('predict.html', feature_names=feature_names)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)