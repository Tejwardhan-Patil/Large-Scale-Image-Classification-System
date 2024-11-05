import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from utils.data_loader import load_data
from utils.metrics import compute_metrics
from models.architectures import cnn, resnet, efficientnet

import json
import os
import numpy as np

def load_model(model_name, model_path):
    """
    Load the model architecture and weights from the specified model name and path.

    Args:
        model_name (str): The name of the model architecture (cnn, resnet, efficientnet).
        model_path (str): The path to the saved model weights.

    Returns:
        torch.nn.Module: The loaded model with the architecture and weights.
    """
    if model_name == 'cnn':
        model = cnn.CNNModel()
    elif model_name == 'resnet':
        model = resnet.ResNet()
    elif model_name == 'efficientnet':
        model = efficientnet.EfficientNet()
    else:
        raise ValueError(f"Model {model_name} not supported.")
    
    # Load the model's state dictionary (weights)
    model.load_state_dict(torch.load(model_path))
    model.eval()  # Set model to evaluation mode
    return model

def evaluate(model, dataloader, device):
    """
    Evaluate the model on the test dataset.

    Args:
        model (torch.nn.Module): The model to be evaluated.
        dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (str): Device on which to perform evaluation ('cpu' or 'cuda').

    Returns:
        float: The average loss over the test dataset.
        dict: The computed metrics (accuracy, precision, recall, F1-score, etc.).
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    
    # Disable gradient calculation during evaluation
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Get the predicted classes
            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Calculate average loss
    avg_loss = running_loss / len(dataloader)
    
    # Compute evaluation metrics
    metrics = compute_metrics(all_labels, all_preds)
    
    return avg_loss, metrics

def save_results(results, output_path):
    """
    Save the evaluation results to a JSON file.

    Args:
        results (dict): Dictionary of evaluation results (loss and metrics).
        output_path (str): File path to save the results.
    """
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=4)

def load_test_data(data_path, batch_size):
    """
    Load the test dataset for evaluation.

    Args:
        data_path (str): The directory path containing the test data.
        batch_size (int): Batch size for loading the data.

    Returns:
        DataLoader: DataLoader for the test dataset.
    """
    dataset = load_data(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    return dataloader

def parse_config(config_path):
    """
    Parse the evaluation configuration file.

    Args:
        config_path (str): Path to the configuration file (JSON or YAML).

    Returns:
        dict: Parsed configuration as a dictionary.
    """
    with open(config_path, 'r') as f:
        config = json.load(f)
    return config

def print_metrics(metrics):
    """
    Print the evaluation metrics in a formatted way.

    Args:
        metrics (dict): Dictionary containing evaluation metrics.
    """
    print("Evaluation Metrics:")
    for metric, value in metrics.items():
        print(f"{metric.capitalize()}: {value:.4f}")

def main():
    """
    Main function to execute the evaluation process.
    """
    # Load configuration file
    config_path = 'configs/evaluation_config.json'
    config = parse_config(config_path)

    # Load test data
    test_loader = load_test_data(config['data_path'], config['batch_size'])

    # Load the model
    model = load_model(config['model_name'], config['model_path'])

    # Evaluate the model
    loss, metrics = evaluate(model, test_loader, config['device'])

    # Print the evaluation results
    print(f"Evaluation Loss: {loss:.4f}")
    print_metrics(metrics)

    # Save the results to a file
    results = {
        'loss': loss,
        'metrics': metrics
    }
    save_results(results, config['output_path'])


if __name__ == "__main__":
    main()

def log_evaluation_details(model_name, loss, metrics, log_path):
    """
    Log the evaluation details such as loss and metrics to a specified file.

    Args:
        model_name (str): The name of the evaluated model.
        loss (float): The average evaluation loss.
        metrics (dict): Dictionary of evaluation metrics (accuracy, precision, etc.).
        log_path (str): Path to the log file.
    """
    log_data = {
        'model_name': model_name,
        'loss': loss,
        'metrics': metrics
    }

    # Append log data to the log file
    with open(log_path, 'a') as log_file:
        log_file.write(json.dumps(log_data, indent=4))
        log_file.write("\n\n")


def compute_confusion_matrix(all_labels, all_preds, num_classes):
    """
    Compute the confusion matrix for the evaluation.

    Args:
        all_labels (list): List of ground truth labels.
        all_preds (list): List of predicted labels.
        num_classes (int): The number of classes in the dataset.

    Returns:
        np.array: A confusion matrix representing the model's predictions vs actual labels.
    """
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    
    for true_label, pred_label in zip(all_labels, all_preds):
        confusion_matrix[true_label][pred_label] += 1
    
    return confusion_matrix


def plot_confusion_matrix(confusion_matrix, class_names, output_path):
    """
    Plot the confusion matrix and save it as an image.

    Args:
        confusion_matrix (np.array): Confusion matrix to plot.
        class_names (list): List of class names corresponding to the labels.
        output_path (str): Path to save the plotted confusion matrix image.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')

    # Save the confusion matrix plot to the specified output path
    plt.savefig(output_path)
    plt.close()


def calculate_classification_report(all_labels, all_preds, class_names):
    """
    Generate a classification report with precision, recall, and F1-score for each class.

    Args:
        all_labels (list): List of ground truth labels.
        all_preds (list): List of predicted labels.
        class_names (list): List of class names.

    Returns:
        dict: Classification report with precision, recall, and F1-score for each class.
    """
    from sklearn.metrics import classification_report

    report = classification_report(all_labels, all_preds, target_names=class_names, output_dict=True)
    return report


def print_classification_report(classification_report):
    """
    Print the classification report in a formatted way.

    Args:
        classification_report (dict): The classification report to be printed.
    """
    print("Classification Report:")
    for class_name, metrics in classification_report.items():
        if class_name != 'accuracy':
            precision = metrics['precision']
            recall = metrics['recall']
            f1_score = metrics['f1-score']
            print(f"Class: {class_name}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1_score:.4f}")
        else:
            print(f"Overall Accuracy: {metrics:.4f}")


def load_class_names(class_names_path):
    """
    Load class names from a file.

    Args:
        class_names_path (str): Path to the file containing the class names.

    Returns:
        list: List of class names.
    """
    with open(class_names_path, 'r') as f:
        class_names = [line.strip() for line in f.readlines()]
    return class_names


def extended_evaluation(model, dataloader, device, num_classes, class_names, output_dir):
    """
    Perform an extended evaluation, including confusion matrix and classification report.

    Args:
        model (torch.nn.Module): The model to evaluate.
        dataloader (torch.utils.data.DataLoader): DataLoader for the test dataset.
        device (str): Device to perform evaluation on ('cpu' or 'cuda').
        num_classes (int): The number of classes in the dataset.
        class_names (list): List of class names.
        output_dir (str): Directory to save evaluation outputs (confusion matrix, classification report).
    
    Returns:
        dict: A dictionary with loss, metrics, confusion matrix, and classification report.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    running_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = running_loss / len(dataloader)
    metrics = compute_metrics(all_labels, all_preds)

    # Compute confusion matrix
    confusion_matrix = compute_confusion_matrix(all_labels, all_preds, num_classes)

    # Plot and save confusion matrix
    confusion_matrix_path = os.path.join(output_dir, 'confusion_matrix.png')
    plot_confusion_matrix(confusion_matrix, class_names, confusion_matrix_path)

    # Generate classification report
    classification_report = calculate_classification_report(all_labels, all_preds, class_names)
    
    # Save classification report to JSON file
    classification_report_path = os.path.join(output_dir, 'classification_report.json')
    with open(classification_report_path, 'w') as f:
        json.dump(classification_report, f, indent=4)

    # Return extended evaluation results
    results = {
        'loss': avg_loss,
        'metrics': metrics,
        'confusion_matrix': confusion_matrix.tolist(),
        'classification_report': classification_report
    }
    return results


def run_extended_evaluation(config):
    """
    Run extended evaluation based on configuration settings.

    Args:
        config (dict): Configuration settings for extended evaluation.
    """
    # Load test data
    test_loader = load_test_data(config['data_path'], config['batch_size'])

    # Load class names
    class_names = load_class_names(config['class_names_path'])
    num_classes = len(class_names)

    # Load the model
    model = load_model(config['model_name'], config['model_path'])

    # Perform extended evaluation
    results = extended_evaluation(
        model,
        test_loader,
        config['device'],
        num_classes,
        class_names,
        config['output_dir']
    )

    # Log the evaluation results
    log_evaluation_details(config['model_name'], results['loss'], results['metrics'], config['log_path'])

    # Print evaluation results
    print(f"Extended Evaluation Loss: {results['loss']:.4f}")
    print_metrics(results['metrics'])
    print_classification_report(results['classification_report'])

    # Return the evaluation results
    return results

def export_results_to_csv(results, output_csv_path):
    """
    Export evaluation results including metrics and classification report to a CSV file.

    Args:
        results (dict): Dictionary containing evaluation metrics and classification report.
        output_csv_path (str): Path to the CSV file to export results.
    """
    import csv

    # Open the CSV file for writing
    with open(output_csv_path, mode='w', newline='') as file:
        writer = csv.writer(file)

        # Write the overall metrics
        writer.writerow(['Metric', 'Value'])
        for metric, value in results['metrics'].items():
            writer.writerow([metric.capitalize(), value])

        writer.writerow([])  # Blank line for separation

        # Write the classification report
        writer.writerow(['Class', 'Precision', 'Recall', 'F1-Score'])
        for class_name, metrics in results['classification_report'].items():
            if class_name != 'accuracy':
                precision = metrics['precision']
                recall = metrics['recall']
                f1_score = metrics['f1-score']
                writer.writerow([class_name, precision, recall, f1_score])

    print(f"Results exported to {output_csv_path}")


def visualize_class_distribution(dataloader, class_names, output_path):
    """
    Visualize and save the class distribution of the dataset used for evaluation.

    Args:
        dataloader (torch.utils.data.DataLoader): DataLoader for the dataset.
        class_names (list): List of class names.
        output_path (str): Path to save the class distribution plot.
    """
    import matplotlib.pyplot as plt

    # Initialize a counter for each class
    class_counts = {class_name: 0 for class_name in class_names}

    # Iterate through the dataset to count each class occurrence
    for _, labels in dataloader:
        for label in labels:
            class_counts[class_names[label]] += 1

    # Create a bar plot for class distribution
    plt.figure(figsize=(12, 6))
    plt.bar(class_counts.keys(), class_counts.values(), color='skyblue')
    plt.xlabel('Classes')
    plt.ylabel('Frequency')
    plt.title('Class Distribution in the Dataset')

    # Rotate x-ticks for readability
    plt.xticks(rotation=45, ha='right')

    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()

    print(f"Class distribution plot saved at {output_path}")


def run_full_evaluation(config):
    """
    Run the full evaluation process including extended evaluation, result export, and visualization.

    Args:
        config (dict): Configuration settings for full evaluation.
    """
    # Run extended evaluation
    results = run_extended_evaluation(config)

    # Export results to CSV
    export_results_to_csv(results, config['csv_output_path'])

    # Visualize class distribution
    test_loader = load_test_data(config['data_path'], config['batch_size'])
    class_names = load_class_names(config['class_names_path'])
    visualize_class_distribution(test_loader, class_names, config['class_distribution_output_path'])

    print("Full evaluation completed successfully.")


def schedule_evaluation(run_interval_hours, config):
    """
    Schedule periodic evaluations based on a specified time interval.

    Args:
        run_interval_hours (int): The time interval in hours between evaluations.
        config (dict): Configuration settings for evaluation.
    """
    import time

    while True:
        print(f"Starting evaluation at {time.strftime('%Y-%m-%d %H:%M:%S')}")

        # Run the full evaluation
        run_full_evaluation(config)

        # Wait for the specified interval before the next evaluation
        print(f"Next evaluation in {run_interval_hours} hours.")
        time.sleep(run_interval_hours * 3600)  # Convert hours to seconds


def load_config(config_file_path):
    """
    Load the configuration from a specified file path.

    Args:
        config_file_path (str): Path to the configuration file (JSON format).

    Returns:
        dict: Configuration settings.
    """
    with open(config_file_path, 'r') as f:
        config = json.load(f)
    return config


def parse_arguments():
    """
    Parse command-line arguments for running evaluations.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    import argparse

    parser = argparse.ArgumentParser(description='Run model evaluation.')

    parser.add_argument('--config', type=str, required=True, help='Path to the configuration file.')
    parser.add_argument('--schedule', type=int, default=None, help='Interval (in hours) to run periodic evaluations.')

    return parser.parse_args()


def main():
    """
    Main function to execute the evaluation process with optional scheduling.
    """
    # Parse command-line arguments
    args = parse_arguments()

    # Load configuration from file
    config = load_config(args.config)

    # Check if evaluation should be scheduled
    if args.schedule:
        print(f"Scheduling evaluations every {args.schedule} hours.")
        schedule_evaluation(args.schedule, config)
    else:
        # Run the full evaluation process once
        run_full_evaluation(config)


if __name__ == "__main__":
    main()