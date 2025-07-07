## Fake News Prediction using Logistic Regression

_This project aims to classify news articles as **real** or **fake** using Logistic Regression — one of the most interpretable machine learning models. The dataset contains labeled news headlines and text, and the model is trained on preprocessed and vectorized input features._

#### Dataset Details
- Source: Kaggle
- Files Used: `train.csv`
- **Columns:**
  - `title`: Headline of the news article  
  - `text`: Full news article content  
  - `author` : Author of the article
  - `label`: `1` for fake news, `0` for real news

#### ML Workflow: 
1. Importing Libraries
    1. `numpy`, `pandas` for data handling  
    2. `sklearn` for ML and evaluation  
    3. `nltk` for text preprocessing  
    4. `re` for regular expressions 
2. Data Preprocessing
    1. Combined `title` and `text` features  
    2. Removed null values  
    3. Converted text to lowercase  
    4. Removed punctuation and special characters using regex
3. Stemming
    1. Used `nltk.PorterStemmer` to reduce words to their root forms  
4. Text Vectorization
    1. Used `TF-IDF Vectorizer` to convert text into numerical features  
    2. Removed stop words
5.  Model Training
    1. Split data into training and test sets (80:20)  
    2. Trained a `Logistic Regression` model on the vectorized features
6. Evaluation
    1. Metrics used: `Accuracy`, `Precision`, `Recall`

#### Results
Train Accuracy: 0.986
Test Accuracy: 0.977
![Screenshot 2025-07-07 202453](https://github.com/user-attachments/assets/5ecd61d3-1103-4d98-ab29-4f3294dde641)
![Screenshot 2025-07-07 202500](https://github.com/user-attachments/assets/5a9da63f-b798-4806-ab88-7f04fffba9aa)

#### What I Learned
1. Text preprocessing: stemming, cleaning, and normalization
2. How to vectorize text using TF-IDF
3. How Logistic Regression can be used for binary classification
4. Evaluating NLP models using classification metrics
