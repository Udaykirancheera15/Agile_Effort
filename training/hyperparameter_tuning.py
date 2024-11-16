import optuna
from models.lstm2vec import create_lstm2vec_model
from models.model_utils import prepare_text_data
from data.preprocess import prepare_data
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import EarlyStopping
import mlflow

# Load data
X, y = prepare_data('path/to/dataset.csv')
X_context, vocab_size, tokenizer = prepare_text_data(X['context'])
X_train, X_val, y_train, y_val = train_test_split(X_context, y, test_size=0.2, random_state=42)

def objective(trial):
    # Define hyperparameters to tune
    embedding_dim = trial.suggest_int('embedding_dim', 50, 200)
    lstm_units = trial.suggest_int('lstm_units', 32, 128)
    learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-2)
    
    # Initialize model with trial parameters
    model = create_lstm2vec_model(vocab_size=vocab_size, embedding_dim=embedding_dim, input_length=X_train.shape[1])
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss='mean_squared_error', metrics=['mae'])

    # Early stopping for better tuning performance
    early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

    # Train the model with the current set of hyperparameters
    history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping], verbose=0)
    
    # Evaluate model on validation set and return validation MAE as the metric to minimize
    val_mae = min(history.history['val_mae'])
    return val_mae

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)  # Adjust n_trials as needed for more exhaustive search

    # Print the best parameters and their performance
    print("Best parameters:", study.best_params)
    print("Best validation MAE:", study.best_value)

    mlflow.set_experiment("Hyperparameter Tuning - LSTM2Vec")

    with mlflow.start_run():
        study = optuna.create_study(direction='minimize')

        # Callback to log each trial
        def log_trial(trial):
            mlflow.log_params(trial.params)
            mlflow.log_metric("val_mae", trial.value)

        study.optimize(objective, n_trials=50, callbacks=[log_trial])

        # Log best parameters and their performance
        mlflow.log_params(study.best_params)
        mlflow.log_metric("best_val_mae", study.best_value)
        print("Best parameters:", study.best_params)
        print("Best validation MAE:", study.best_value)

