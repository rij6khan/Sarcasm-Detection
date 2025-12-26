## CS 461 - Final Project

Project Members:
Rija Khan (rk1047)
Kelly Xu (nx27)

Contents of this project:
* experiments: the 2 models that were created for experimentation
  * glove+BiLSTM: the neural network model tested
  * TF-IDF+linearSVM: the linear SVM model tested
* models: model weights loaded to predict_sarcasm.py
* predict_sarcasm.py: sarcasm prediction file that makes predictions and runs analysis on the test.csv file given
* readme.txt: this file
* report.pdf: our report
* requirements.txt: all required libraries/dependencies needed to run the file
* test.csv: the CSV file provided for testing


How to run predict_sarcasm.py:
As the project description says, run the predict_sarcasm.py file with this command in the terminal:
```python predict_sarcasm.py --input [input CSV file] --output predictions.csv```