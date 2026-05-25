# Spam Message Classifier using NLP and Naive Bayes

A beginner machine learning project that classifies SMS messages as spam or ham using Natural Language Processing (NLP) techniques with Python and scikit-learn.

## Features

* Loads and processes a spam message dataset
* Cleans and prepares text data
* Converts text into numerical features using CountVectorizer
* Uses a Naive Bayes classifier for prediction
* Splits data into training and testing sets
* Evaluates model performance using accuracy metrics
* Predicts custom spam messages

## Technologies Used

* Python
* pandas
* scikit-learn

## Machine Learning Concepts

* Natural Language Processing (NLP)
* Text Classification
* Bag of Words
* Naive Bayes Classification
* Train/Test Split

## How to Run

```bash
pip install -r requirements.txt
python main.py
```

## Example Prediction

Input:

```python
message = ["You won free money"]
```

Output:

```python
Spam
```

## Accuracy

The model achieves around 95–99% accuracy depending on the dataset split.

## Future Improvements

* Add TF-IDF vectorization
* Add confusion matrix visualization
* Build a Streamlit web application
* Compare multiple machine learning models
* Deploy the model online

## Dataset

The dataset was obtained from Kaggle.

## Inspiration

Inspired by:
[https://medium.com/@RobuRishabh/natural-language-processing-linear-text-classification-898f64a1e04a](https://medium.com/@RobuRishabh/natural-language-processing-linear-text-classification-898f64a1e04a)
