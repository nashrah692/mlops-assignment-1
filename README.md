# mlops-assignment-1

## PROBLEM STATEMENT
Build, compare, monitor and register a classification model for the Iris dataset. The task is to train multiple models, evaluate and compare them with standard classification metrics, track experiments with MLflow, and register the best model in the MLflow Model Registry.

## DATASET SELECTION
I used the IRIS dataset from scikit-learn:
  - 150 samples and 3 classes
  - Features:
      - Sepal Length
      - Sepal Width
      - Petal Length
      - Petal Width
  - Loaded in code using sklearn.datasets.load_iris()

## MODEL SELECTION & COMPARISON
The following three models were selected:
  - Logistic Regression
  - Random Forest
  - XGBoost
Used the following metrics for comaprison:
   - Accuracy
   - Precision
   - F1 score
   - Recall
   - Confusion Matrix

## MLFlow LOGGING SCREENSHOTS
<img width="1819" height="526" alt="image" src="https://github.com/user-attachments/assets/13bbbca9-91c0-492f-8771-621cde5844de" />
<img width="1834" height="755" alt="image" src="https://github.com/user-attachments/assets/b19032c6-0ff3-45d8-a286-f15a1d425026" />
<img width="1402" height="410" alt="image" src="https://github.com/user-attachments/assets/d36fc790-1229-4852-bb64-e4fab3021419" />
<img width="1637" height="463" alt="image" src="https://github.com/user-attachments/assets/90fa5c8e-1881-4ecd-905d-494d9802d8ab" />
<img width="1700" height="858" alt="image" src="https://github.com/user-attachments/assets/d126b4c6-0cda-43cd-b6ca-be2c0f030f3b" />

## MLFLow REGISTRATION SCREENSHOTS
<img width="1540" height="278" alt="image" src="https://github.com/user-attachments/assets/5a0bf4f7-c19b-4921-9a99-2d3cd3df1603" />

## INSTRUCTIONS TO RUN THE CODE
Run the train.py from the src folder using the following command:
python src/train.py

Run the register_model.py from the src folder using the following command:
python src/register_model.py

## MODEL REGISTRATION STEPS
After training and logging the models on MLFlow I compared the performance of the models. Then based on the best accuracy model I selected the model to register.

The best model was registered on MLFlow with version number 3

Code:
model_uri = f"runs:/<best_run_id>/model"
mlflow.register_model(model_uri=model_uri, name="iris-classifier")

## COMPLETE STEPS EXPLAINED 

I am using PyCharm, Github and MLFlow to load a dataset named "IRIS" and then train, compare, monitor and register models.

## PART 1
  - Created a Github Repository.
  - Cloned the repository in PyCharm.
  - Created folders that were to be used in the process.
  - Created a virtual environment.
  - Installed dependencies.
  - Made an Initial Commit on Github.

## PART 2
  - Loaded a dataset: "IRIS"
  - Trained 3 models on this dataset:
      - Logistic Regression
      - Random Forest
      - XGBoost
  - Evaluated the models.
  - Saved the models in the models folder.

## PART 3
  - Logged the trained models to MLFlow.
  - Also logged their evaluation metrics.
  - Compared the models based on the following metrics:
      - F1 score
      - Accuracy
      - Recall
      - Precision
   
## PART 4
  - Selected the best performing model based on accuracy.
  - After selecting the best model registered it on MLFlow with a version number.
