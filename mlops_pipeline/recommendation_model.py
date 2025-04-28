import joblib
import mlflow.pyfunc
import tensorflow as tf

class RecommendationModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        self.model = tf.keras.models.load_model("./models/model_16.keras")

    def predict(self, context, model_input):
        # Predict next 24 values for each row in model_input
        future_preds = self.model.predict(model_input)  # shape: (n_samples, 24)
        print(future_preds)
        scaled_predicted_price = future_preds[-1]
        scaled_current_price = model_input[-1][-1][-1]  # Or wherever your current price is

        scaler = joblib.load('../data/scalers/Close_scaler.pkl')

        predicted_price = scaler.inverse_transform([[scaled_predicted_price]])
        current_price = scaler.inverse_transform([[scaled_current_price]])

        threshold = 0.0005  # Set your threshold for buy/sell

        def recommend_by_threshold(predicted_price, current_price):
            delta = (predicted_price - current_price) / current_price

            if delta > threshold:
                return 'long'
            elif delta < -threshold:
                return 'short'
            else:
                return 'hold'

        response = {
            'prediction': future_preds,
            'recommendation': recommend_by_threshold(predicted_price, current_price)
        }
        return response
