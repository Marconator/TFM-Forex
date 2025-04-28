import mlflow

from recommendation_model import RecommendationModel

artifacts = {"model_path": "./models/model_16.keras"}

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Forex-Prediction_CNN-LSTM")

mlflow.pyfunc.log_model(
    artifact_path="eurusd_rec_model",
    python_model=RecommendationModel(),
    artifacts=artifacts
)