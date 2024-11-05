import optuna
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.architectures.cnn import CNN
from data_loader import CustomDataset
import logging
import os
from datetime import datetime

# Setup logging
log_dir = 'logs/hyperparameter_tuning'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

logging.basicConfig(
    filename=os.path.join(log_dir, f'optuna_tuning_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(message)s',
)

# Hyperparameter tuning utility class for better organization
class HyperparameterTuner:
    def __init__(self, model_class, data_class, device='cpu'):
        self.model_class = model_class
        self.data_class = data_class
        self.device = device

    def load_data(self, batch_size):
        train_dataset = self.data_class(split='train')
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        val_dataset = self.data_class(split='val')
        val_loader = DataLoader(val_dataset, batch_size=batch_size)

        return train_loader, val_loader

    def build_model(self, trial):
        num_layers = trial.suggest_int('num_layers', 1, 5)
        model = self.model_class(num_layers=num_layers)
        return model

    def select_optimizer(self, trial, model):
        learning_rate = trial.suggest_loguniform('learning_rate', 1e-5, 1e-1)
        optimizer_name = trial.suggest_categorical('optimizer', ['Adam', 'RMSprop', 'SGD'])

        if optimizer_name == 'Adam':
            optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        elif optimizer_name == 'RMSprop':
            optimizer = optim.RMSprop(model.parameters(), lr=learning_rate)
        else:
            optimizer = optim.SGD(model.parameters(), lr=learning_rate)

        return optimizer

    def train(self, trial, model, train_loader, val_loader):
        criterion = nn.CrossEntropyLoss()
        optimizer = self.select_optimizer(trial, model)
        model.to(self.device)
        
        num_epochs = 10
        best_val_accuracy = 0.0

        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0
            for batch in train_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            val_loss, val_accuracy = self.validate(model, val_loader, criterion)
            trial.report(val_accuracy, epoch)
            logging.info(f"Epoch {epoch + 1}/{num_epochs}, Loss: {running_loss:.4f}, Val Acc: {val_accuracy:.4f}")

            if val_accuracy > best_val_accuracy:
                best_val_accuracy = val_accuracy

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

        return best_val_accuracy

    def validate(self, model, val_loader, criterion):
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch
                inputs, labels = inputs.to(self.device), labels.to(self.device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        val_accuracy = correct / total
        logging.info(f"Validation Loss: {val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        return val_loss, val_accuracy


# Define the objective function for Optuna
def objective(trial):
    tuner = HyperparameterTuner(CNN, CustomDataset, device='cuda' if torch.cuda.is_available() else 'cpu')
    
    # Hyperparameters
    batch_size = trial.suggest_categorical('batch_size', [16, 32, 64, 128])

    # Load data
    train_loader, val_loader = tuner.load_data(batch_size)

    # Build model
    model = tuner.build_model(trial)

    # Train model
    accuracy = tuner.train(trial, model, train_loader, val_loader)

    return accuracy

# Callback function to handle trial completion logging
def trial_callback(study, trial):
    logging.info(f"Trial {trial.number} completed. Value: {trial.value}, Params: {trial.params}")

# Main entry point for Optuna study
if __name__ == "__main__":
    logging.info("Starting hyperparameter tuning...")

    study_name = "cnn_hyperparam_tuning"  # Unique identifier for the study
    study = optuna.create_study(direction="maximize", study_name=study_name)

    study.optimize(objective, n_trials=50, callbacks=[trial_callback])

    # Save best hyperparameters
    logging.info(f"Best hyperparameters: {study.best_params}")
    print(f"Best hyperparameters: {study.best_params}")

    # Save the study results for future analysis
    study_path = f'{log_dir}/{study_name}_study.pkl'
    with open(study_path, 'wb') as f:
        import pickle
        pickle.dump(study, f)

    logging.info(f"Study saved to {study_path}")

# Hyperparameter search space exploration
def explore_hyperparameter_space(study):
    best_trial = study.best_trial
    logging.info(f"Best Trial: Value={best_trial.value}")
    for key, value in best_trial.params.items():
        logging.info(f"  {key}: {value}")

    trials = study.trials_dataframe()
    logging.info(f"All trials:\n{trials}")
    return trials

# Advanced hyperparameter suggestion strategies
def apply_advanced_sampling(trial):
    # Adaptive TPE or another custom sampling strategy
    return trial.suggest_loguniform('advanced_param', 1e-4, 1e-2)


# Model checkpointing and result saving
def save_model_checkpoint(model, checkpoint_dir='checkpoints/'):
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    model_path = os.path.join(checkpoint_dir, f"model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pth")
    torch.save(model.state_dict(), model_path)
    logging.info(f"Model checkpoint saved at {model_path}")


# Visualization (loss curves) - Utility function
def visualize_study_results(study):
    import matplotlib.pyplot as plt

    trials = study.trials_dataframe()
    plt.plot(trials['number'], trials['value'])
    plt.xlabel('Trial')
    plt.ylabel('Accuracy')
    plt.title('Optuna Hyperparameter Tuning Progress')
    plt.grid()
    plot_path = os.path.join(log_dir, f"optuna_study_progress_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plt.savefig(plot_path)
    logging.info(f"Study result plot saved at {plot_path}")