
import logging
from pathlib import Path
import mlflow

import joblib
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_validate,
    GridSearchCV
)

from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier

from src.features.build_features import create_tfidf_vectorizer


# --------------------------------------------------
# 1. Logging configuration
# --------------------------------------------------

logger = logging.getLogger(__name__)
mlflow.set_experiment('reddit sentiment analysis')

# --------------------------------------------------
# 2. Training function
# --------------------------------------------------

def train_model():

    logger.info("Starting model training workflow")

    # --------------------------------------------------
    # 3. Define project paths
    # --------------------------------------------------

    project_root = Path(__file__).resolve().parents[2]

    processed_data_path = (
        project_root / "data" / "processed" / "reddit_cleaned.csv"
    )

    # --------------------------------------------------
    # 4. Load cleaned dataset
    # --------------------------------------------------

    df = pd.read_csv(processed_data_path)

    logger.info("Cleaned data loaded successfully")
    logger.info("Dataset shape: %s", df.shape)

    # --------------------------------------------------
    # 5. Separate features and target
    # --------------------------------------------------

    X = df["clean_comment"]
    y = df["category"]

    logger.info("Features and target separated")

    # --------------------------------------------------
    # 6. Train-test split
    # --------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    logger.info("Train-test split completed")
    logger.info("Training samples: %s", len(X_train))
    logger.info("Testing samples: %s", len(X_test))

    # --------------------------------------------------
    # 7. Save test data
    # --------------------------------------------------

    test_data = pd.DataFrame({
        "clean_comment": X_test,
        "category": y_test
    })

    test_data_path = (
        project_root / "data" / "processed" / "test_data.csv"
    )

    test_data.to_csv(test_data_path, index=False)

    logger.info("Test data saved at: %s", test_data_path)

    # --------------------------------------------------
    # 8. Cross-validation strategy
    # --------------------------------------------------

    cv = StratifiedKFold(
        n_splits=5,
        shuffle=True,
        random_state=42
    )

    # --------------------------------------------------
    # 9. Evaluation metrics
    # --------------------------------------------------

    scoring = {
        "accuracy": "accuracy",
        "precision_macro": "precision_macro",
        "recall_macro": "recall_macro",
        "f1_macro": "f1_macro"
    }

    # --------------------------------------------------
    # 10. Define candidate models
    # --------------------------------------------------

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000
        ),

        "Multinomial Naive Bayes": MultinomialNB(),

        "Linear SVM": LinearSVC(
            random_state=42
        ),

        "SGD Classifier": SGDClassifier(
            random_state=42
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            random_state=42,
            n_jobs=1
        )
    }

    # --------------------------------------------------
    # 11. Compare models using Cross-Validation
    # --------------------------------------------------

    model_results = {}

    for model_name, classifier in models.items():

        logger.info("Evaluating model: %s", model_name)

        pipeline = Pipeline([
            ("tfidf", create_tfidf_vectorizer()),
            ("classifier", classifier)
        ])
        with mlflow.start_run(run_name=model_name):
            cv_results = cross_validate(
            estimator=pipeline,
            X=X_train,
            y=y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=True
        )
            mlflow.log_param('model_name',model_name)
            mlflow.log_param('cv_strategy','StratifiedKFold')
            mlflow.log_param('n_splits',5)

            classifier_params = classifier.get_params()
            for param_name ,param_value in classifier_params.items():
                mlflow.log_param(
                    f'classifier_{param_name}',
                    str(param_value)
                )

        # Log cross-validation metrics
            for metric in scoring:
                scores = cv_results[f"test_{metric}"]
                mean_score=scores.mean()
                std_score=scores.std()
                mlflow.log_metric(f'cv_{metric}_mean',mean_score)
                mlflow.log_metric(f'cv_{metric}_std',std_score)
                logger.info(
                "%s - %s: %.4f (+/- %.4f)",
                model_name,
                metric,
                scores.mean(),
                scores.std()
            )

        # Store model results
            model_results[model_name] = {
            "pipeline": pipeline,
            "mean_f1": cv_results["test_f1_macro"].mean(),
            "std_f1": cv_results["test_f1_macro"].std(),
            "cv_results": cv_results
        }

    # --------------------------------------------------
    # 12. Select the best model
    # --------------------------------------------------

    best_model_name = max(
        model_results,
        key=lambda name: model_results[name]["mean_f1"]
    )

    best_pipeline = model_results[best_model_name]["pipeline"]

    logger.info("Best model selected: %s", best_model_name)

    logger.info(
        "Best Cross-Validation F1: %.4f",
        model_results[best_model_name]["mean_f1"]
    )

    # --------------------------------------------------
    # 13. Hyperparameter tuning
    # --------------------------------------------------

    param_grids = {

        "Logistic Regression": {
            "classifier__C": [0.1, 1, 10],
            "classifier__class_weight": [None, "balanced"]
        },

        "Multinomial Naive Bayes": {
            "classifier__alpha": [0.1, 0.5, 1.0, 2.0]
        },

        "Linear SVM": {
            "classifier__C": [0.01, 0.1, 1, 10],
            "classifier__class_weight": [None, "balanced"]
        },

        "SGD Classifier": {
            "classifier__loss": ["hinge", "log_loss"],
            "classifier__alpha": [0.0001, 0.001, 0.01],
            "classifier__class_weight": [None, "balanced"]
        },

        "Random Forest": {
            "classifier__n_estimators": [100, 200],
            "classifier__max_depth": [None, 20, 40]
        }
    }

    logger.info("Starting GridSearchCV")

    grid_search = GridSearchCV(
        estimator=best_pipeline,
        param_grid=param_grids[best_model_name],
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
        refit=True
    )
    with mlflow.start_run(run_name=f"{best_model_name}_GridSearch"):
        mlflow.log_param("selected_model", best_model_name)
        mlflow.log_param("tuning_method", "GridSearchCV")
        mlflow.log_param("scoring", "f1_macro")
        grid_search.fit(X_train, y_train)

        for param_name , param_value in grid_search.best_params_.items():
            mlflow.log_param(
            f"best_{param_name}",
            param_value
        )

    # 4. Log best CV score
            mlflow.log_metric(
        "best_cv_f1_macro",
        grid_search.best_score_
    )

    logger.info("GridSearchCV completed")

    logger.info(
        "Best parameters: %s",
        grid_search.best_params_
    )

    logger.info(
        "Best GridSearch CV F1: %.4f",
        grid_search.best_score_
    )

    # --------------------------------------------------
    # 14. Get the best trained model
    # --------------------------------------------------

    best_model = grid_search.best_estimator_

    # --------------------------------------------------
    # 15. Save the best model
    # --------------------------------------------------

    model_dir = project_root / "artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / "best_model.joblib"

    joblib.dump(best_model, model_path)

    logger.info("Best model saved at: %s", model_path)

    # --------------------------------------------------
    # 16. Return training results
    # --------------------------------------------------

    logger.info("Model training workflow completed")

    return {
        "best_model": best_model,
        "best_model_name": best_model_name,
        "model_results": model_results,
        "grid_search": grid_search,
        "X_test": X_test,
        "y_test": y_test
    }


# --------------------------------------------------
# 17. Main execution
# --------------------------------------------------

if __name__ == "__main__":

    train_model()
