import pandas as pd
import numpy as np
import spacy
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

nlp = spacy.load("en_core_web_sm")

def load_and_preprocess_data(file_path):
    # Load dataset
    df = pd.read_csv(file_path, parse_dates=['created'])
    
    # Handle missing values
    df['context'] = df['context'].fillna('')
    df['codesnippet'] = df['codesnippet'].fillna('')
    
    # Extract temporal features from 'created'
    df['year'] = df['created'].dt.year
    df['month'] = df['created'].dt.month
    df['day_of_week'] = df['created'].dt.dayofweek
    df.drop(columns=['created', 'issuekey'], inplace=True)
    
    # Define feature and target columns
    X = df.drop(columns='storypoint')
    y = df['storypoint']
    
    return X, y

def build_preprocessing_pipeline(df):
    text_features = ['context', 'codesnippet']
    numeric_features = ['year', 'month', 'day_of_week']
    categorical_features = ['t_Story', 't_Technical.task', 't_Bug', 't_Improvement', 't_Epic'] + [col for col in df.columns if col.startswith('c_')]
    preprocessor = ColumnTransformer(
        transformers=[
            ('context_tfidf', TfidfVectorizer(ngram_range=(1, 2)), 'context'),
            ('code_tfidf', TfidfVectorizer(ngram_range=(1, 2)), 'codesnippet'),
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(), categorical_features)
        ]
    )
    return preprocessor

def prepare_data(file_path):
    X, y = load_and_preprocess_data(file_path)
    preprocessor = build_preprocessing_pipeline(X)
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    # Fit-transform on training data
    X_train = preprocessor.fit_transform(X_train)
    X_val = preprocessor.transform(X_val)
    return X_train, X_val, y_train, y_val, preprocessor
