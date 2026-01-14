import mlflow
from mlflow_integration import train_model  # Импортируем функцию из mlflow_integration.py

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("Emotion-Classification-FineTuning")

# Эксперимент с разными learning rates
learning_rates = [1e-5, 2e-5, 5e-5]
for lr in learning_rates:
    with mlflow.start_run(nested=True):  # Nested runs для подэкспериментов
        mlflow.log_param("learning_rate", lr)
        results = train_model(learning_rate=lr)
        mlflow.log_metrics(results)

print("Эксперимент по подбору learning rate завершен!")