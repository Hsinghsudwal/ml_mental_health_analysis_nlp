# Ml Mental Health Analysis Nlp

![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)

## Table of Content
- [Problem Statement](#problem-statement)
- [Setup](#setup)
- [Development](#development)
- [Orchestration](#orchestration)
- [Deployment](#deployment)
- [Monitoring](#monitoring)
- [Best Practice](#best-practice)
- [CICD](#cicd)

## Problem Statement
Mental health issues affect people worldwide, but many individuals don't get diagnosed or seek help when they need it most. Traditional ways of diagnosing, like self-reporting or clinical evaluations, can sometimes be slow or inaccurate. 

This project aims to build a machine learning model that can classify text into seven categories of mental health: Normal, Depression, Suicidal, Anxiety, Stress, Bi-Polar, and Personality Disorder. By using Natural Language Processing (NLP), the model will analyze language and emotional cues in text, helping to spot early signs of mental health issues. The goal is to provide quicker, more accurate insights that can help improve support and intervention for those who need it.

**Content:** Mental health conditions often manifest through changes in language use, emotional tone, and patterns in communication. By analyzing textual data, we can uncover these patterns to help identify various mental health issues early. The dataset used for this project consists of labeled text data, with each sample corresponding to one of seven mental health categories.

***Mental Health Categories:***

* Normal: This category includes individuals expressing a balanced emotional state, without signs of distress or mental health issues.
* Depression: Texts from individuals showing symptoms of depression such as sadness, hopelessness, and withdrawal.
* Suicidal: Texts indicating a potential risk of self-harm or suicidal thoughts.
* Anxiety: Expresses worry, fear, and restlessness, often associated with anxiety disorders.
* Stress: Texts reflecting excessive stress or pressure, potentially from work, family, or personal issues.
* Bi-Polar: Symptoms alternating between extreme high-energy periods (mania) and depressive states.
* Personality Disorder: Exhibits patterns of thoughts and behaviors that are inconsistent or disruptive to daily functioning.

**Objective:** Develop a text classification model capable of accurately predicting the mental health category of a given text input.  Explore NLP techniques and machine learning models, such as TF-IDF and CountVector for feature extraction and Random Forest, SVM, MultinomialNB or LightGBM for classification. Implement strategies for improving model accuracy, handling imbalanced data, and generalizing the model to real-world text inputs.

#### Data Source: The dataset for this project is sourced from [Kaggle](https://www.kaggle.com/datasets/suchintikasarkar/sentiment-analysis-for-mental-health)
In order to work locally make sure download the data and store it in `data/data_name.csv`

## Setup

1. **Installation:** Clone the repository git clone https://github.com/Hsinghsudwal/ml_mental_health_analysis_nlp.git

2. Set up Environment for managing libraries and running python scripts.

    ```bash
    conda create -n venv python==3.12 -y
    ``` 
    activate the environment `conda activate venv`

3. Install Dependencies: `pip install -r requirements.txt`


## Development

**Notebook:** Open terminal from your main project directory `cd notebook` and run `jupyter notebook`

**Solution:**
1. Data Pre-processing: Is to remove unwanted characters, punctuation, stop words, and unnecessary spaces. Applying Tokenization and Lemmatization/ Stemming: Words will be broken down into tokens and reduced to their root forms to ensure consistency in the data.
Handling Imbalanced Data: Since mental health issues like depression or suicidal thoughts may be underrepresented in real-world text data, techniques such as SMOTE (Synthetic Minority Over-sampling Technique).

2. Feature Engineering: TF-IDF (Term Frequency-Inverse Document Frequency): This technique will be applied to convert the raw text data into numerical features by measuring the importance of each word in the context of the entire dataset.

3. Model Selection and Training: Traditional Machine Learning Models: Start with Random Forest, SVM (Support Vector Machine), Logistic Regression and etc. to evaluate their performance.

4. Evaluation Metrics: The model will be evaluated using metrics such as accuracy, precision, recall, F1-score, and confusion matrix. These metrics are essential to understand how well the model performs, especially considering the importance of minimizing false negatives (missing out on mental health issues).


## Orchestration

**Steps:**
1. Run Python Scripts: From your main project directory, open terminal and run your code editor. convert your jupyter notebook to scripts via `jupyter nbconvert --to scripts notebook.ipynb`

2. Create modular code to perform pipeline functions. These files are located: 
* `src/project_name/data_loader.py` which will load data from sources, in this case, a `.csv` file. 
* `src/project_name/data_transformation.py` This script handles transformations such as pre-processing, feature engineering and splitting the dataset into training and test.
* `src/project_name/model_trainer.py`This script trains the model using lightgbm. Also (joblib) save model and the class_ labels.
* `src/project_name/model_evaluation.py` This script will evaluate model metrics such as accuracy, f1.

**Locally:** How to run pipeline: On terimal of the project directory run `python run.py`.

**LocalStack:** Run LocalStack using the following command:

    ```bash
    docker-compose up --build -d
    ```

`docker-compose down` to kill docker

Train and Upload the Model S3:
when LocalStack is running in the background, you can train the model and upload it to S3 and access LocalStack services at: `http://localhost:4566`

Train model
```bash
python run.py
```
Upload the model to the LocalStack S3 bucket:
```bash
python upload_s3.py
```
Interacting with LocalStack:
* First AWS CLI by configuring it to point to LocalStack.
```bash
aws --endpoint-url=http://localhost:4566 s3 ls -[list of all s3]
and
aws --endpoint-url=http://localhost:4566 s3 ls <bucket-name> -[content of bucket]
```

## Deployment

Model Deployment: From your main project directory `cd deployment` and create application name with `app.py` and `dockerfile` for deployment. The app is for Streamlit Chatbot: A web-based application will be created using Streamlit to deploy by asking user to use S3 cloud trainer model or local trainer model. This app will allow users to input text and get predictions on the mental health state.

```bash
streamlit run app.py
```

**Docker**
1. build docker image
```bash
docker build -t my-app .
```
2. Running docker
```bash
docker run -p 8501:8501 my-app
```

**localstack**
When localstack is running we can run in browser streamlit app with this url search:
```bash
http://localhost:8501
```

## Monitoring
From the main project directory, navigate to the monitor directory:
```bash
cd monitor
touch monitor_app.py
```
**Streamlit:** Is a python framework for building interactive web apps and create dashboards layout
First create app script to perform model metrics on:
* Deploy a trained machine learning model.
* Track key model metrics (accuracy, precision, recall, F1 score).
* Compare the tracked metrics with predefined thresholds.
* Alert if the model’s performance is dropping, indicating the need for retraining.

```bash
streamlit run monitor_app.py
```
**docker**
1. build docker image
```bash
docker build -t monitor-app .
```
2. Running docker
```bash
docker run -p 8501:8501 monitor-app
```

**localstack**
```bash
http://localhost:8502
```
#### Note: Running monitoring locally run `streamlit run monitor_app.py` which works well, but with testing *locastack* it need to upload that temporary data
## Best Practice
Cloud Deployment (Optional):
AWS S3: using LocalStack for testing

**Test**: pytest

**Format**: Black *.py

**Makefile**: `make format`, python: `make main`, docker: `make localstack`, bucket: `make s3`, streamlit-app: `make deploy`, streamlit-monitor: `make monitor`
            

## CICD
github_action pipeline

### Access the apps

1. On main project directory
2. `python run.py`
3. `docker-compose up --build`
4. `python upload_s3.py`
5. For deployment Streamlit app will be available at `http://localhost:8501`
6. For monitoring streamlit app is available at `http://localhost:8502`
7. LocalStack is available at `http://localhost:4566`.




















