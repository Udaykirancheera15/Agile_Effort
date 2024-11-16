from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, LSTM, Dense
from tensorflow.keras.optimizers import Adam

def create_lstm2vec_model(vocab_size, embedding_dim=50, input_length=512):
    # Define the input layer
    input_text = Input(shape=(input_length,), name='input_text')
    
    # Embedding layer to map each word to a vector
    embedding = Embedding(input_dim=vocab_size, output_dim=embedding_dim, input_length=input_length)(input_text)
    
    # LSTM layer to encode sequence information
    lstm_output = LSTM(50, name='lstm_layer')(embedding)
    
    # Output dense layer for regression
    dense_output = Dense(1, activation='linear', name='output')(lstm_output)
    
    # Define and compile model
    model = Model(inputs=input_text, outputs=dense_output)
    model.compile(optimizer=Adam(learning_rate=1e-3), loss='mean_squared_error', metrics=['mae'])
    
    return model

