# ICT_report_code_Xudong
Overview:
This project aims to train a multimodal spam classification model. According to the file categories processed, there are 4 models for classifying text spam, namely GCNN, STC, CNN_LSTM, BERT_spam, and one model for classifying image spam, namely CNN_Spam_Image_Classification
 
Features:
Data preprocessing (cleaning, tokenization, vectorization)
Model training and evaluation


Requirements:
Python 3.10.2
torch  2.1.2
numpy 1.25.2
matplotlib 3.6.2


How to run?
The file path for training GCNN model and STC model is: ./text_spam_classification/GCNN_STC.py
The file path for training CNN_LSTM model is: ./text_spam_classification/GCNN_STC.py
The file path for training BERT_based_Spam_Classification model and evaluating performance is: ./text_spam_classification/BERT_spam_filtering.ipynb (this needs to be opened in colab)
The path for storing text email content classification model files is: ./text_spam_classification/my_model

The file path for training image spam classification model is: ./Image_spam_classification/CNN_Spam_Image_Classification.py
The path for storing image email content classification model files is: ./Image_spam_classification/myModel

The file path for overall evaluation of model performance is: ./performance_evaluation/for_Performance_Evaluation.py



