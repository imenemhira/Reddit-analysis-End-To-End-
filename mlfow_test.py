import mlflow

mlflow.set_experiment("Sentiment Analysis")

with mlflow.start_run():

    mlflow.log_param("model", "Logistic Regression")

    mlflow.log_param("max_iter", 1000)

    mlflow.log_metric("f1_score", 0.91)