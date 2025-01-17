# SMS Spam Detection System Using NLP and Python

## Project Overview
This project implements an SMS Spam Detection System using Natural Language Processing (NLP) techniques in Python. The system classifies text messages as either **"spam"** or **"ham"** (non-spam). By leveraging machine learning models and text preprocessing methods, the project demonstrates efficient text classification to filter unwanted messages.

## Features
- **Spam and Ham Classification**: The system predicts whether a message is spam or not.
- **Text Preprocessing**: Cleans and transforms text data using stemming and lemmatization techniques.
- **Vectorization Techniques**: Utilizes both CountVectorizer and TF-IDF Vectorizer for feature extraction.
- **Multiple Models**: Demonstrates different approaches for text processing, including:
  - Using **Porter Stemmer** with **TF-IDF Vectorizer**
  - Using **Lemmatizer** with **CountVectorizer**
  - Using **Stemmer** with **CountVectorizer**

## Prerequisites
- Python 3.x
- Required libraries:
  - pandas
  - numpy
  - sklearn
  - nltk

## Installation
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/sms-spam-detection.git
   cd sms-spam-detection
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Project Structure
```
.
├── sms_using_Stemmer_with_countvectorizer.ipynb
├── sms_using_lemmatizer_with_countvectorizer.ipynb
├── sms_using_PorterStemmer_with_TFIdf_Vectorizer.ipynb
├── README.md
├── requirements.txt
└── data
    └── sms_spam_collection.csv  # Dataset file
```

## Dataset
The SMS Spam Collection dataset contains labeled messages for training and testing.
- **Label**: Indicates if a message is spam or ham.
- **Message**: The content of the SMS.

Example:
```
spam	Free entry in a weekly competition to win...
ham	Hey, what are you doing?
```

## NLP Techniques and Approach
1. **Data Preprocessing**:
   - Convert text to lowercase.
   - Remove punctuation and special characters.
   - Tokenize and apply stemming or lemmatization.
2. **Feature Extraction**:
   - **CountVectorizer**: Converts text into a bag-of-words representation.
   - **TF-IDF Vectorizer**: Weighs terms based on their importance in the corpus.
3. **Model Training**:
   - Uses **Naive Bayes** or other classifiers to predict spam vs. ham.

## Usage
Run the respective notebooks to see the implementation:
```bash
jupyter notebook sms_using_Stemmer_with_countvectorizer.ipynb
jupyter notebook sms_using_lemmatizer_with_countvectorizer.ipynb
jupyter notebook sms_using_PorterStemmer_with_TFIdf_Vectorizer.ipynb
```

## Results
- The system achieves high accuracy in distinguishing spam from ham messages.
- Comparative performance of different preprocessing and vectorization techniques can be analyzed from the notebooks.

## Future Improvements
- Incorporate deep learning models for improved classification.
- Expand dataset for better generalization.

## License
This project is licensed under the MIT License.

---

Thank you for using the SMS Spam Detection System! Feel free to contribute and enhance this project.

