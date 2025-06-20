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


---

## **How to do**

### ✅ Step 1: Prepare Datasets for Training Models

- **1-1.** Before running `CNN_LSTM.py` and `GCNN_STC.py` for text spam classification, please:
  - Download the dataset file.
  - Unzip it.
  - Place the unzipped file named `DataSet_shuffle.csv` in the `./text_spam_classification/` directory.
  - _(Note: The `BERT_spam_filtering.ipynb` model only needs to be run in **Colab**.)_

- **1-2.** Before running `CNN_Spam_Image_Classification.py` for image spam classification, please:
  - Download the image dataset.
  - Unzip it.
  - Place the folder named `DataSet_small` in the `./Image_spam_classification/` directory.

---

### ✅ Step 2: Prepare for Evaluation

- Before running `for_Performance_Evaluation.py` for performance evaluation, please:
  - Download the model weight files.
  - Unzip the archive.
  - Place the four model weight files into the `./performance_evaluation/` directory.



