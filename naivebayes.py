# Solaiman Ibrahimi - ML/AI Personal Project | COMPLETED :: 5/20/23 |
# Implemented an Email Spam detector using the Principles of Text Classification & 
# The Naive Bayes Algorithm
# Backend takes input email headers then displays a Metric Evaluation in the front-end GUI
# Tools used: Numpy, Pandas, PyQt(Frontend GUI)


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



# This function's job is to go through the files at a given path and return 
# the emails contained in the files.
def readFiles(path):
    for root, dirnames, filenames in os.walk(path):
        for filename in filenames:
            path = os.path.join(root, filename)
            inBody = False
            lines = [] 
            f = io.open(path, 'r', encoding='latin1')
            for line in f:
                if inBody:
                    lines.append(line)
                elif line == '\n':
                    inBody = True
                    
            f.close()
            message = '\n'.join(lines)
            # YIELD HERE...
            yield path, message

# Function here takes emails from above
def dataFrameFromDirectory(path, classification):
    rows = []
    index = []
    for filename, message in readFiles(path):
        rows.append({'message': message, 'class': classification})
        index.append(filename)
    return DataFrame(rows, index=index)


data = DataFrame({'message': [], 'class': []})
# Tester input...
my_emails = [
    {'message': 'Hello, this is a legitimate email.', 'class': 'ham'},
    {'message': 'Get a discount on our products. Limited time offer!', 'class': 'spam'},
    {'message': 'You sent $12 to Shams Wardak', 'class': 'ham'},
    {'message': 'SOME0NEE MAY HAVE RUN A BACKGROUND-CHECK ON YOUU', 'class': 'spam'},
    {'message': 'Insurance Requirement Information For Union Blacksburg', 'class': 'ham'},
    {'message': 'YOU HAVE RECEIVED A PAYMENT OF 1000 USD', 'class': 'spam'},
]


for email in my_emails:
    data = data._append(email, ignore_index=True)

    
# Using vectors to split messages into organized lists
vectorizer = CountVectorizer()
counts = vectorizer.fit_transform(data['message'].values)

# Classes used for probability computation later in Metric Evaluation
targets = data['class'].values

classifier = MultinomialNB()
classifier.fit(counts, targets)

# Text sampler
sample = ['50 lbs in 61 days: New No-Exercise Skinny Pill Melts Belly Fat', "Cliplar: DRUSKI and Mike | Rxqe posted 2 new videos", "SOLAIMAN: Applicants Requested"]

sample_counts = vectorizer.transform(sample)

proba_predictions = classifier.predict_proba(sample_counts)

# Displays in IDE output for easy testing 
for i, email in enumerate(sample):
    print(f"\nEmail: {email}")
    print("Spam or Ham probability:")
    for j, class_label in enumerate(classifier.classes_):
        print(f"{class_label}: {proba_predictions[i][j]}")


# Metric Evaluator
predicted_targets = cross_val_predict(classifier, counts, targets, cv=2)

# Calculation of evaluation metrics
accuracy = accuracy_score(targets, predicted_targets)
precision = precision_score(targets, predicted_targets, pos_label='spam')
recall = recall_score(targets, predicted_targets, pos_label='spam')
f1 = f1_score(targets, predicted_targets, pos_label='spam')

# GUI Handler (Frontend STARTS HERE).........................................................
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

        self.detect_button = QPushButton("Detect Spam", self)
        layout.addWidget(self.detect_button)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        self.detect_button.clicked.connect(self.detect_spam)

    def detect_spam(self):
        sample = ['50 lbs in 61 days: New No-Exercise Skinny Pill Melts Belly Fat', "Cliplar: DRUSKI and Mike | Rxqe posted 2 new videos", "SOLAIMAN: Applicants Requested"]
        sample_counts = vectorizer.transform(sample)
        proba_predictions = classifier.predict_proba(sample_counts)

        results = ""
        for i, email in enumerate(sample):
            results += f"Email: {email}\nSpam or Ham probability:\n"
            for j, class_label in enumerate(classifier.classes_):
                results += f"{class_label}: {proba_predictions[i][j]}\n"
            results += "\n"
            
        predicted_targets = cross_val_predict(classifier, counts, targets, cv=2)

        accuracy = accuracy_score(targets, predicted_targets)
        precision = precision_score(targets, predicted_targets, pos_label='spam')
        recall = recall_score(targets, predicted_targets, pos_label='spam')
        f1 = f1_score(targets, predicted_targets, pos_label='spam')

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

