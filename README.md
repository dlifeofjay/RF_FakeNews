Fake News Detection Using Machine Learning
Overview

This project aims to build a machine learning model capable of identifying fake news and real news articles. The model utilizes pattern recognition in the text to differentiate between real news and fake news based on various features such as word usage, sentence structure, and linguistic patterns. The classifier has been trained using a Random Forest model, a robust ensemble learning technique that combines multiple decision trees to improve prediction accuracy.

Project Details

Classification Task: The goal is to classify news articles into two categories:

Real News (1)

Fake News (0)

Model Used: Random Forest Classifier

Random Forest is a powerful machine learning algorithm that creates multiple decision trees and aggregates their predictions to produce a final output. This method reduces overfitting and improves model performance compared to individual decision trees.

Input Data: The model takes textual news articles as input and extracts relevant features based on word frequency, word patterns, and syntactic structures within the text.

Output: The model outputs either a 1 (Real News) or 0 (Fake News) based on the classification.

Key Features

Pattern Recognition: The model identifies specific patterns in words and text structure that differentiate fake news from real news.

Random Forest Classifier: Leveraging ensemble learning for better prediction accuracy and robustness against overfitting.

Feature Engineering: Features such as word counts, n-grams, sentiment analysis, and lexical diversity are used to enhance model performance.

Installation

To run the project locally, follow the steps below:

Clone the repository:

git clone https://github.com/dlifeofjay/RF_FakeNews
cd RF_FakeNews


Install required dependencies:

pip install -r requirements.txt


Run the model:

streamlit run RF_app.py

Model Performance

The model’s performance is evaluated using common metrics such as accuracy, precision, recall, and F1-score.

For more details on the performance evaluation and results, please refer to the PDF documentation provided with this repository.

File Structure

RF_app.py: Main script for running the model using streamlit.

requirements.txt: List of required Python libraries.

RF_FakeNews.ipynb: file where model was trained.

RFFN_model.joblib: Directory for saving the trained model.

RFFN_vectorizer.joblib: Directory for saving the saved vectorizer

Additional Information

For further insights into the structure of the model, its performance evaluation, and how it was developed, please refer to the PDF documentation file included in the repository. The document provides a comprehensive explanation of the methodology, results, and possible improvements.
