import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")

experiment_name = "Iris Classification"
experiment = mlflow.get_experiment_by_name(experiment_name)

# sorting by best accuracy
client = MlflowClient()
runs = client.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy DESC"],
    max_results=1
)

best_run = runs[0]
print("Best Run ID:", best_run.info.run_id)
print("Best Accuracy:", best_run.data.metrics["accuracy"])

# registering the model
model_name = "iris-classifier"
model_uri = f"runs:/{best_run.info.run_id}/model"

mlflow.register_model(model_uri=model_uri, name=model_name)
print(f"Model has been registered.\n Model: {model_name} from run {best_run.info.run_id}")