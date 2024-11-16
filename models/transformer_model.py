from transformers import TFAutoModel, AutoTokenizer
import tensorflow as tf

def create_transformer_model():
    # Load transformer and tokenizer
    transformer = TFAutoModel.from_pretrained('distilbert-base-uncased')
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    # Define input layers
    input_ids = tf.keras.Input(shape=(512,), dtype='int32', name='input_ids')
    attention_mask = tf.keras.Input(shape=(512,), dtype='int32', name='attention_mask')
    
    # Pass inputs through the transformer model
    embeddings = transformer(input_ids, attention_mask=attention_mask)[0]
    
    # Use the CLS token output for regression
    output = tf.keras.layers.Dense(1, activation='linear')(embeddings[:, 0, :])  # CLS token output
    
    # Compile model
    model = tf.keras.Model(inputs=[input_ids, attention_mask], outputs=output)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5), loss='mean_squared_error', metrics=['mae'])
    
    return model, tokenizer

