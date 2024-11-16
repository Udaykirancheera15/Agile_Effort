from data.preprocess import prepare_data
from models.lstm2vec import create_lstm2vec_model
from models.model_utils import prepare_text_data
from sklearn.model_selection import train_test_split
import numpy as np
import mlflow
import mlflow.keras

# Load and preprocess data
X_train, X_val, y_train, y_val, preprocessor = prepare_data('data/datasets/Tawosi_Dataset/XD_tfidf-se.csv')
X_context, vocab_size, context_vectorizer = prepare_text_data(X_train['context'])
X_code, code_vocab_size, code_vectorizer = prepare_text_data(X_train['codesnippet'])

X_val_context, val_context_vocab_size, val_context_vectorizer = prepare_text_data(X_val['context'])
X_val_code, val_code_vocab_size, val_code_vectorizer = prepare_text_data(X_val['codesnippet'])

# Initialize and train LSTM2Vec model
model = create_lstm2vec_model(context_vocab_size=vocab_size, code_vocab_size=code_vocab_size)
history = model.fit(
    [X_context, X_code],
    y_train,
    validation_data=([X_val_context, X_val_code], y_val),
    epochs=10,
    batch_size=32
)

# Set up MLflow
mlflow.set_experiment("AgileEffortEstimation")
with mlflow.start_run():
    mlflow.log_param("model_type", "LSTM2Vec")
    mlflow.log_param("embedding_dim", 50)
    mlflow.log_param("context_vocab_size", vocab_size)
    mlflow.log_param("code_vocab_size", code_vocab_size)
    # Log training metrics
    history = model.fit(
        [X_context, X_code],
        y_train,
        validation_data=([X_val_context, X_val_code], y_val),
        epochs=5,
        batch_size=32
    )
    for epoch, metrics in enumerate(history.history['mae']):
        mlflow.log_metric("mae", metrics, step=epoch)
    # Log and save the model
    mlflow.keras.log_model(model, "lstm2vec_model")
