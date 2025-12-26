## Rutgers CS 461 – Final Project
## Sarcasm Detection

This project explores sarcasm detection using both neural and classical machine learning approaches. We evaluate a GloVe-based BiLSTM model and compare it against a TF-IDF + Linear SVM baseline.

### Project Members
* Rija Khan
* Kelly Xu

### Project Structure
- **experiments**: Models created for experimentation
  - **glove+BiLSTM**: Neural network model
  - **TF-IDF+linearSVM**: Linear SVM baseline
- **models**: Saved model weights used by `predict_sarcasm.py`
- **predict_sarcasm.py**: Runs sarcasm prediction and evaluation on a given CSV file
- **readme.txt**: This file
- **report.pdf**: Final project report
- **requirements.txt**: Required Python dependencies
- **train.csv**: Training dataset
- **valid.csv**: Validation dataset
- **test.csv**: Test dataset

### How to Run `predict_sarcasm.py`

First, install the required dependencies:

```pip install -r requirements.txt```

Then, run the prediction script as described in the project specification:

```python predict_sarcasm.py --input [input_csv_file] --output predictions.csv```
