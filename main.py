# Import libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# Load the spam dataset from CSV file
Spam_data = pd.read_csv(
    "data/spam.csv",
    encoding="latin-1",
    usecols=[0,1],
    names=["labels", "messages"],
    header=0
)

print(Spam_data.head())

# Count how many spam and ham messages exist
print(Spam_data["labels"].value_counts())

# X contains the text messages 
X = Spam_data["messages"]

# Y contains the labels (spam or ham)
Y = Spam_data["labels"]

# convert text into numerical data
vectorizer = CountVectorizer()
X=vectorizer.fit_transform(X)

# Split data into training and testing sets | 80% training and 20% testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

# Create the Naive Bayes model
model= MultinomialNB()
# Train the model using training data
model.fit(X_train, Y_train) 

# Predict labels for test data
predictions = model.predict(X_test)
print("Prediction:", predictions[0])

# Calculate and print model accuracy
print("Accuracy:", round(accuracy_score(Y_test,predictions),2))


# Adding new message to test prediction
message = ["You won free vacation"]
prediction = model.predict(vectorizer.transform(message))

print("Message:", message[0])
print("Prediction:", prediction[0])


