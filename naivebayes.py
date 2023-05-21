# Naive Bayes
import os
import io
import sys
import numpy as np
import pandas as pd
from pandas import DataFrame
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import cross_val_predict
import sys
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton



# This function does the heavy lifting in this assignment. It's job is to go through the files at a given path and return 
# the emails contained in the files.
def readFiles(path):
    # this is how we iterate across files at 'path'
    for root, dirnames, filenames in os.walk(path):
        # because our route only has files...
        for filename in filenames:
            # absolute path to file
            path = os.path.join(root, filename)

            # TODO_1: initialize the flag that is used in the loop below to distinguish between 'header' and 'body' 
            inBody = False
            #this is where the lines from email body will be saved.
            lines = []
            # opening current file for reading. The 'r' param means read access. 
            f = io.open(path, 'r', encoding='latin1')
            # reading one line at a time
            # HINT: look at the emails manually and notice what separates the header and body content.
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
   
                    
            # after the loop is finished, close the file.
            f.close()
            # goes through each string and combines into a big strink separated with spaces.
            message = '\n'.join(lines)
            # TODO_4: research the difference between 'yield' and 'return' to understand why we use yield here.
            yield path, message

# This function relies on the function above. Here, we grab the emails from the above function and 
# place them into individual data frames (you can think of it as if it is a table of JSONs where each JSON has an email plus its 
# classification)
def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    
 
    for filename, message in readFiles(path):
        
        rows.append({'message': message, 'class': classification})
        index.append(filename)

    # data frame object takes two arrays 'rows'=emails, and 'index'=filenames
    return DataFrame(rows, index=index)

# A DataFrame is a convenient class that allows you to create a table-like structure. 
# In our case we are trying to have a column with the messages and a column that classifies the type
# of the message.
data = DataFrame({'message': [], 'class': []})

# Including the email details with the spam/ham classification in the dataframe
# TODO_0: you must specify the path of the unzipped folders 
# Replace <path> based on where these directories are relative to this script
my_emails = [
    {'message': 'Hello, this is a legitimate email.', 'class': 'ham'},
    {'message': 'Get a discount on our products. Limited time offer!', 'class': 'spam'},
    {'message': 'You sent $12 to Shams Wardak', 'class': 'ham'},
    {'message': 'SOME0NEE MAY HAVE RUN A BACKGROUND-CHECK ON YOUU', 'class': 'spam'},
    {'message': 'Insurance Requirement Information For Union Blacksburg', 'class': 'ham'},
    {'message': 'YOU HAVE RECEIVED A PAYMENT OF 1000 USD', 'class': 'spam'},
    
    # Add more emails as needed
]
for email in my_emails:
    data = data._append(email, ignore_index=True)


# and print the content of the data frames, note you can also print the head and tail of the data
# TODO_3: Try this with the full dataset.

# ***this code should be added at the bottom****





#CountVectorizer is used to split up each message into its list of words
#Then we throw them to a MultinomialNB classifier function from scikit
#2 inputs required: actual data we are training on and the target data
vectorizer = CountVectorizer()

# vectorizer.fit_trsnform computes the word count in the emails and represents that as a frequency matrix (e.g., 'free' occured 1304 times.)
counts = vectorizer.fit_transform(data['message'].values)

#we will need to also have a list of ham/spam (corresponding to the emails from 'counts') that will allow Bayes Naive classifier compute the probabilities.
targets = data['class'].values

# This is from the sklearn package. MultinomialNB stands for Multinomial Naive Bayes classsifier
classifier = MultinomialNB()
# when we feed it the word frequencies plus the spam/ham mappings, the classifier will create a table of probabilities similar ot the one that you saw in the first assignment in this module.
classifier.fit(counts, targets)

#Time to have fun! You can compute P(ham| email text) and P(spam | email text) using classifier.predict(...emails...) 
#... but in what format should we supply the emails we want to test?

# first, transform this list into a table of word frequencies.

# after that you are ready to do the predictions.


sample = ['50 lbs in 61 days: New No-Exercise Skinny Pill Melts Belly Fat', "Cliplar: DRUSKI and Mike | Rxqe posted 2 new videos", "SOLAIMAN: Applicants Requested"]

sample_counts = vectorizer.transform(sample)

proba_predictions = classifier.predict_proba(sample_counts)

for i, email in enumerate(sample):
    print(f"\nEmail: {email}")
    print("Spam or Ham probability:")
    for j, class_label in enumerate(classifier.classes_):
        print(f"{class_label}: {proba_predictions[i][j]}")



#Model Evaluator
predicted_targets = cross_val_predict(classifier, counts, targets, cv=2)

# Calculate evaluation metrics
accuracy = accuracy_score(targets, predicted_targets)
precision = precision_score(targets, predicted_targets, pos_label='spam')
recall = recall_score(targets, predicted_targets, pos_label='spam')
f1 = f1_score(targets, predicted_targets, pos_label='spam')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Spam Detection")
        self.setGeometry(100, 100, 400, 400)

        # Create a QVBoxLayout to arrange the widgets
        layout = QVBoxLayout()

        # Create a QLabel to display the output
        self.output_label = QLabel(self)
        self.output_label.setWordWrap(True)  # Enable text wrapping
        layout.addWidget(self.output_label)

        # Create a QPushButton to trigger the spam detection
        self.detect_button = QPushButton("Detect Spam", self)
        layout.addWidget(self.detect_button)

        # Create a QWidget to hold the layout
        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.detect_button.clicked.connect(self.detect_spam)

    def detect_spam(self):
        # Perform spam detection here using your existing code
        sample = ['50 lbs in 61 days: New No-Exercise Skinny Pill Melts Belly Fat', "Cliplar: DRUSKI and Mike | Rxqe posted 2 new videos", "SOLAIMAN: Applicants Requested"]
        sample_counts = vectorizer.transform(sample)
        proba_predictions = classifier.predict_proba(sample_counts)

        # Format the results as a string
        results = ""
        for i, email in enumerate(sample):
            results += f"Email: {email}\nSpam or Ham probability:\n"
            for j, class_label in enumerate(classifier.classes_):
                results += f"{class_label}: {proba_predictions[i][j]}\n"
            results += "\n"

        # Perform model evaluation
        predicted_targets = cross_val_predict(classifier, counts, targets, cv=2)

        # Calculate evaluation metrics
        accuracy = accuracy_score(targets, predicted_targets)
        precision = precision_score(targets, predicted_targets, pos_label='spam')
        recall = recall_score(targets, predicted_targets, pos_label='spam')
        f1 = f1_score(targets, predicted_targets, pos_label='spam')

        # Display the model evaluation results
        results += "\nModel Evaluation:\n"
        results += f"Accuracy: {accuracy:.4f}\n"
        results += f"Precision: {precision:.4f}\n"
        results += f"Recall: {recall:.4f}\n"
        results += f"F1 Score: {f1:.4f}\n"

        self.output_label.setText(results)

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()

