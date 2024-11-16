import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

def prepare_text_data(text_data):
    # Preprocess the text data
    text_data = text_data.apply(lambda x: re.sub(r'[^a-zA-Z0-9\s]', '', str(x)).lower())
    text_data = text_data.apply(lambda x: ' '.join([word for word in x.split() if word not in stopwords.words('english')]))
    text_data = text_data.apply(lambda x: ' '.join([PorterStemmer().stem(word) for word in x.split()]))
    
    # Tokenize the text data
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text_data)
    text_sequences = tokenizer.texts_to_sequences(text_data)
    
    return text_sequences, len(tokenizer.word_index) + 1, tokenizer
