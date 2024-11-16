import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from models.lstm2vec import create_lstm2vec_model
from models.model_utils import prepare_text_data
from data.preprocess import prepare_data
from sklearn.model_selection import train_test_split
import joblib
import tensorflow as tf

# Load data
X, y = prepare_data('path/to/dataset.csv')
X_context, vocab_size, tokenizer = prepare_text_data(X['context'])

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_context, y, test_size=0.2, random_state=42)

# Retrieve best hyperparameters from Optuna study
best_params = study.best_params  # Replace with loaded parameters if needed

# Initialize and train model with best parameters
model = create_lstm2vec_model(vocab_size=vocab_size, embedding_dim=best_params['embedding_dim'])
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=best_params['learning_rate']), loss='mean_squared_error')
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Evaluate on test set
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Test Mean Squared Error: {mse}")
print(f"Test R^2 Score: {r2}")

# Save model
model.save("results/model_output/best_lstm2vec_model.h5")

# Save tokenizer
joblib.dump(tokenizer, "results/model_output/tokenizer.joblib")

