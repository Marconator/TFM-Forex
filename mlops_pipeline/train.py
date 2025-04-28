from keras.src.callbacks import EarlyStopping
import os
import mlflow
import numpy as np
import pandas as pd
from mlflow.models import infer_signature

from model_utils import (create_dataset, cnn_lstm_model,
                         directional_accuracy)
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import ParameterSampler
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Forex-Prediction_CNN-LSTM")

LOOK_BACK = 60
FORECAST = 24
WINDOW_SIZE = 10000
TEST_SIZE = 1000
N_ITER = 27
N_FEATURES = 12
DA_WEIGHT = 0.2
constants = { 'LOOK_BACK': LOOK_BACK,
              'FORECAST': FORECAST,
              'WINDOW_SIZE': WINDOW_SIZE,
              'TEST_SIZE': TEST_SIZE,
              'N_ITER': N_ITER,
              'N_FEATURES': N_FEATURES,
              'DA_WEIGHT': DA_WEIGHT,
              }


param_grid = {
    'filters': [64, 128, 256],
    'lstm_units': [50, 70, 80],
    'kernel_size': [3, 5, 7],
    'learning_rate': [0.001, 0.0001],
    'dropout_rate': [0.2, 0.3]
}

random_params = list(ParameterSampler(param_grid, n_iter=N_ITER, random_state=42))

# Load dataset
model_dataset = pd.read_csv("../data/train_scaled.csv", index_col='Date')

best_rmse = float('inf')
best_params = {}
best_model = None
best_da = 0
best_idx = -1
run_data = []

for idx, params in enumerate(random_params):
    with mlflow.start_run(run_name=f"Run {idx+1}") as run:
        run_id = run.info.run_id
        print(f"🔧 Testing params {idx+1}/{N_ITER}: {params}")
        results = []
        directional_accuracies = []
        models = []
        trainX = []


        for i in range(WINDOW_SIZE, len(model_dataset) - TEST_SIZE, WINDOW_SIZE):
            train = model_dataset.iloc[0:i, :]
            test = model_dataset.iloc[i:i + TEST_SIZE, :]

            trainX, trainY = create_dataset(train, LOOK_BACK, FORECAST)
            testX, testY = create_dataset(test, LOOK_BACK, FORECAST)

            trainX = np.reshape(trainX, (trainX.shape[0], trainX.shape[1], N_FEATURES))
            testX = np.reshape(testX, (testX.shape[0], testX.shape[1], N_FEATURES))

            model = cnn_lstm_model(LOOK_BACK, FORECAST, N_FEATURES, **params)

            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            history = model.fit(trainX, trainY, epochs=50, batch_size=32, verbose=0, validation_split=0.2, shuffle=False, callbacks=[early_stopping])

            for epoch in range(len(history.history["loss"])):
                mlflow.log_metric("train_loss", history.history["loss"][epoch], step=epoch)
                mlflow.log_metric("val_loss", history.history["val_loss"][epoch], step=epoch)

            test_preds = model.predict(testX)
            rmse = np.sqrt(mean_squared_error(testY.flatten(), test_preds.flatten()))
            da = directional_accuracy(testY, test_preds)

            trained_epochs = len(model.history.history['loss'])
            mlflow.log_metric("trained_epochs", trained_epochs)

            mlflow.log_params(params)
            mlflow.log_params(constants)
            mlflow.log_metric("rmse", rmse)
            mlflow.log_metric("directional_accuracy", da)

            results.append(rmse)
            directional_accuracies.append(da)
            models.append(model)

        composite_scores = [results[i] - DA_WEIGHT * directional_accuracies[i] for i in range(len(results))]
        selected_idx = np.argmin(composite_scores)
        mlflow.log_metric("composite_score", composite_scores[selected_idx])

        run_model = models[selected_idx]
        final_rmse = results[selected_idx]
        final_da = directional_accuracies[selected_idx]

        os.makedirs("models", exist_ok=True)
        model_path = f"./models/model_{idx+1}.keras"
        run_model.save(model_path)
        mlflow.log_artifact(model_path)

        # Infer the model signature
        signature = infer_signature(trainX, run_model.predict(trainX))

        # Log the model
        model_info = mlflow.sklearn.log_model(
            sk_model=run_model,
            artifact_path=f'modelo_{idx+1}',
            signature=signature,
            input_example=trainX,
            registered_model_name=f'modelo_{idx+1}',
        )

        run_data.append({
            'run_id': run_id,
            'rmse': final_rmse,
            'composite_score': composite_scores[selected_idx]
        })

        if final_rmse < best_rmse:
            best_rmse = final_rmse
            best_params = params
            best_model = run_model
            best_da = final_da


if best_idx != -1:
    best_run_id = run_data[best_idx]['run_id']
    client = MlflowClient()
    client.set_tag(best_run_id, "best_model", "true")

print("\nBest model summary:")
best_model.summary()
print(f"Best RMSE: {best_rmse}")
print(f"Best Parameters: {best_params}")
print(f"Best Directional Accuracy: {best_da}")
