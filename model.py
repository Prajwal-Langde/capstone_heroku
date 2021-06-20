import pandas as pd 
from nltk.corpus import stopwords 
from sklearn.feature_extraction.text import TfidfVectorizer
import joblib
import xgboost 

#Perform text preprocessing on the sentiments of the 20 products recommended by our recommendation system
def clean_reviews(df): 

    df['reviews_text'] = df['reviews_text'].replace(r'<ed>','', regex = True)
    df['reviews_text'] = df['reviews_text'].replace(r'\B<U+.*>|<U+.*>\B|<U+.*>','', regex = True)
    # convert sentiments to lowercase
    df['reviews_text'] = df['reviews_text'].str.lower()    
    #remove user mentions
    df['reviews_text'] = df['reviews_text'].replace(r'^(@\w+)',"", regex=True)    
    #remove_symbols
    df['reviews_text'] = df['reviews_text'].replace(r'[^a-zA-Z0-9]', " ", regex=True)
    #remove punctuations 
    df['reviews_text'] = df['reviews_text'].replace(r'[[]!"#$%\'()\*+,-./:;<=>?^_`{|}]+',"", regex = True)    
    #remove words of length 1 or 2 
    df['reviews_text'] = df['reviews_text'].replace(r'\b[a-zA-Z]{1,2}\b','', regex=True)
    #remove extra spaces in the tweet
    df['reviews_text'] = df['reviews_text'].replace(r'^\s+|\s+$'," ", regex=True)
    
    #Removing the stopwords
    stop_words = set(stopwords.words('english'))
    mystopwords = [stop_words]    
    df['reviews_text'] = df['reviews_text'].apply(lambda x: ' '.join([word for word in x.split() if word not in mystopwords]))

    # change the datatype of sentiments by replacing positive by 1 and negative by 0
    df = df.replace({"Positive":1, "Negative":0})
    
    #Tokenize and vectorize reviews of the recommended items
    word_vectorizer = TfidfVectorizer(
        sublinear_tf=True,
        strip_accents='unicode',
        analyzer='word',
        token_pattern=r'\w{1,}',
        stop_words='english',
        ngram_range=(1, 3), 
        max_features=50000
        )
    all_text = df['reviews_text']
    word_vectorizer.fit(all_text)
    train_word_features = word_vectorizer.transform(all_text)

    #Load the sentiment analysis model and make predictions for these reviews
    model = joblib.load("joblib_model_xgb.pkl")
    preds = model.predict(train_word_features)
    df['preds'] = preds
    
    return df