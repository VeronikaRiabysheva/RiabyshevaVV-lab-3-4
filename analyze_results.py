import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")  
client = MlflowClient()
experiment = client.get_experiment_by_name("Emotion-Classification-FineTuning")
runs = client.search_runs(experiment.experiment_id)

print("Результаты экспериментов:")
for run in runs:
    print(f"Run ID: {run.info.run_id}")
    print(f"Parameters: {run.data.params}")
    print(f"Metrics: {run.data.metrics}")
    print("-" * 50)

best_run = max(runs, key=lambda x: x.data.metrics.get('eval_f1_score', -float('inf')))
if 'eval_f1_score' not in best_run.data.metrics:
    best_run = min(runs, key=lambda x: x.data.metrics.get('eval_loss', float('inf')))

print(f"Лучший запуск: {best_run.info.run_id}")
print(f"Лучшие метрики: {best_run.data.metrics}")