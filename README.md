# ICT_report_code_Xudong

## **Overview**

This project aims to train a multimodal spam classification model.  
According to the file categories processed, there are 4 models for classifying text spam, namely GCNN, STC, CNN_LSTM, BERT_spam, and one model for classifying image spam, namely CNN_Spam_Image_Classification.

---

## **Features**

- Data preprocessing (cleaning, tokenization, vectorization)  
- Model training and evaluation  

---

## **Requirements**

- Python 3.10.2  
- torch  2.1.2  
- numpy 1.25.2  
- matplotlib 3.6.2  

---

## **Code Directory for Training**

- `./text_spam_classification/GCNN_STC.py` — GCNN & STC model  
- `./text_spam_classification/CNN_LSTM.py` — CNN_LSTM model  
- `./text_spam_classification/BERT_spam_filtering.ipynb` — BERT model (run in Colab)    
- `./Image_spam_classification/CNN_Spam_Image_Classification.py` — image spam model  
- `./performance_evaluation/for_Performance_Evaluation.py` — overall performance evaluation script
 
---

## **Model Download Links**

- **Models**
  - Text and Image Spam Classification Models: [[Download link](https://drive.google.com/file/d/1NUalC9_Hz1iq2ZNiXcV0WaIZTlWq5w32/view?usp=drive_link)]
---

## **Dataset Download Links**

- **Text Email Dataset**
  - Processed TXT dataset: [Download link](https://drive.google.com/file/d/1inAE9Uy8ous8wrHxWKQV0xRpNsV42JST/view?usp=drive_link)

- **Image Email Dataset**
  - Processed image spam dataset: [Download link](https://drive.google.com/file/d/1kjqzMVxmwbE0hAMOW7zmQVWoS3UjHA_Q/view?usp=drive_link)

| Dataset       | sum   | ham   | spam  | Training set | Validation set | Test set |
|---------------|-------|-------|-------|---------------|----------------|----------|
| TXT dataset   | 33028 | 15080 | 17948 | 23119         | 4955           | 4954     |
| Image dataset | 3000  | 1509  | 1491  | 2100          | 450            | 450      |





