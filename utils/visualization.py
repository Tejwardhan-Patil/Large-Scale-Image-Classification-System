import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import itertools

def plot_loss_curve(train_losses, val_losses, title="Loss Curve", xlabel="Epoch", ylabel="Loss", save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', color='blue', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', color='red', linewidth=2)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_accuracy_curve(train_accuracies, val_accuracies, title="Accuracy Curve", xlabel="Epoch", ylabel="Accuracy", save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(train_accuracies, label='Training Accuracy', color='green', linewidth=2)
    plt.plot(val_accuracies, label='Validation Accuracy', color='orange', linewidth=2)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(loc="lower right", fontsize=12)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix(conf_matrix, class_names, title="Confusion Matrix", cmap="Blues", normalize=False, save_path=None):
    if normalize:
        conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]
        fmt = '.2f'
    else:
        fmt = 'd'
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt=fmt, cmap=cmap, xticklabels=class_names, yticklabels=class_names, cbar=False)
    plt.title(title, fontsize=16)
    plt.xlabel("Predicted", fontsize=14)
    plt.ylabel("True", fontsize=14)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_class_distribution(labels, class_names, title="Class Distribution", xlabel="Class", ylabel="Frequency", save_path=None):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=labels, order=class_names, palette="viridis")
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.xticks(rotation=45)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()
    
def plot_learning_rate_schedule(lr_values, epochs, title="Learning Rate Schedule", xlabel="Epoch", ylabel="Learning Rate", save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(range(epochs), lr_values, label='Learning Rate', color='purple', linewidth=2)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_precision_recall_curve(precision, recall, title="Precision-Recall Curve", xlabel="Recall", ylabel="Precision", save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(recall, precision, label='Precision-Recall Curve', color='blue', linewidth=2)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_roc_curve(fpr, tpr, title="ROC Curve", xlabel="False Positive Rate", ylabel="True Positive Rate", save_path=None):
    plt.figure(figsize=(10, 6))
    plt.plot(fpr, tpr, label='ROC Curve', color='red', linewidth=2)
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_confusion_matrix_in_grid(conf_matrices, class_names, title="Confusion Matrices", cmap="Blues", grid_size=(2, 2), save_path=None):
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(16, 12))
    axes = axes.ravel()
    
    for idx, conf_matrix in enumerate(conf_matrices):
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap=cmap, ax=axes[idx], xticklabels=class_names, yticklabels=class_names, cbar=False)
        axes[idx].set_title(f"{title} {idx+1}", fontsize=16)
        axes[idx].set_xlabel("Predicted", fontsize=14)
        axes[idx].set_ylabel("True", fontsize=14)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_feature_importances(feature_importances, feature_names, title="Feature Importances", xlabel="Importance", ylabel="Features", save_path=None):
    sorted_idx = np.argsort(feature_importances)
    plt.figure(figsize=(10, 6))
    plt.barh(np.array(feature_names)[sorted_idx], np.array(feature_importances)[sorted_idx], color='teal')
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_image_samples(images, labels, class_names, title="Image Samples", grid_size=(3, 3), save_path=None):
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(10, 10))
    axes = axes.ravel()
    
    for idx in range(grid_size[0] * grid_size[1]):
        axes[idx].imshow(images[idx])
        axes[idx].set_title(f"{class_names[labels[idx]]}", fontsize=12)
        axes[idx].axis('off')
    
    plt.suptitle(title, fontsize=16)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_metrics(history, metrics, title_prefix="Metric Curve", xlabel="Epoch", ylabel="Value", save_path=None):
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        plt.plot(history[metric], label=f'Training {metric}', linewidth=2)
        if f"val_{metric}" in history:
            plt.plot(history[f"val_{metric}"], label=f'Validation {metric}', linewidth=2)
        plt.title(f"{title_prefix} - {metric.capitalize()}", fontsize=16)
        plt.xlabel(xlabel, fontsize=14)
        plt.ylabel(ylabel, fontsize=14)
        plt.legend(loc="best", fontsize=12)
        plt.grid(True)
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        plt.show()

def plot_learning_rate_comparison(lr_schedules, labels, epochs, title="Learning Rate Schedule Comparison", xlabel="Epoch", ylabel="Learning Rate", save_path=None):
    plt.figure(figsize=(10, 6))
    
    for idx, lr_schedule in enumerate(lr_schedules):
        plt.plot(range(epochs), lr_schedule, label=labels[idx], linewidth=2)
    
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(loc="best", fontsize=12)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_multiple_loss_curves(loss_histories, labels, title="Multiple Loss Curves", xlabel="Epoch", ylabel="Loss", save_path=None):
    plt.figure(figsize=(10, 6))
    
    for idx, loss_history in enumerate(loss_histories):
        plt.plot(loss_history, label=labels[idx], linewidth=2)
    
    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(loc="upper right", fontsize=12)
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()