import numpy as np
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns

def accuracy(y_true, y_pred):
    return accuracy_score(y_true, y_pred)

def precision(y_true, y_pred, average='macro'):
    return precision_score(y_true, y_pred, average=average)

def recall(y_true, y_pred, average='macro'):
    return recall_score(y_true, y_pred, average=average)

def f1(y_true, y_pred, average='macro'):
    return f1_score(y_true, y_pred, average=average)

def top_k_accuracy(y_true, y_pred_probs, k=5):
    top_k_preds = np.argsort(y_pred_probs, axis=1)[:, -k:]
    correct = np.any(top_k_preds == np.expand_dims(y_true, axis=-1), axis=-1)
    return np.mean(correct)

def mean_class_accuracy(y_true, y_pred):
    classes = np.unique(y_true)
    acc_per_class = []
    for c in classes:
        class_mask = (y_true == c)
        class_acc = np.mean(y_pred[class_mask] == y_true[class_mask])
        acc_per_class.append(class_acc)
    return np.mean(acc_per_class)

def confusion_matrix(y_true, y_pred, num_classes):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for true_label, pred_label in zip(y_true, y_pred):
        cm[true_label, pred_label] += 1
    return cm

def top_n_error_rate(y_true, y_pred_probs, n=5):
    top_n_preds = np.argsort(y_pred_probs, axis=1)[:, -n:]
    incorrect = np.logical_not(np.any(top_n_preds == np.expand_dims(y_true, axis=-1), axis=-1))
    return np.mean(incorrect)

def plot_confusion_matrix(y_true, y_pred, num_classes, labels=None):
    cm = confusion_matrix(y_true, y_pred, num_classes)
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.show()

def per_class_precision_recall_f1(y_true, y_pred, num_classes):
    precisions, recalls, f1s = [], [], []
    for i in range(num_classes):
        y_true_binary = (y_true == i).astype(int)
        y_pred_binary = (y_pred == i).astype(int)
        precisions.append(precision_score(y_true_binary, y_pred_binary))
        recalls.append(recall_score(y_true_binary, y_pred_binary))
        f1s.append(f1_score(y_true_binary, y_pred_binary))
    return precisions, recalls, f1s

def plot_class_performance(y_true, y_pred, num_classes, labels=None):
    precisions, recalls, f1s = per_class_precision_recall_f1(y_true, y_pred, num_classes)
    x = np.arange(num_classes)
    
    plt.figure(figsize=(12, 8))
    plt.bar(x - 0.2, precisions, 0.2, label='Precision')
    plt.bar(x, recalls, 0.2, label='Recall')
    plt.bar(x + 0.2, f1s, 0.2, label='F1-score')
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.xticks(x, labels)
    plt.title('Per-Class Performance')
    plt.legend()
    plt.show()

def roc_auc_multiclass(y_true, y_pred_probs, num_classes):
    roc_auc = []
    for i in range(num_classes):
        y_true_binary = (y_true == i).astype(int)
        roc_auc.append(roc_auc_score(y_true_binary, y_pred_probs[:, i]))
    return roc_auc

def plot_roc_curves(y_true, y_pred_probs, num_classes, labels=None):
    from sklearn.metrics import roc_curve, auc
    plt.figure(figsize=(10, 7))
    
    for i in range(num_classes):
        y_true_binary = (y_true == i).astype(int)
        fpr, tpr, _ = roc_curve(y_true_binary, y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {labels[i]} (AUC = {roc_auc:.2f})')
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Multiclass Classification')
    plt.legend(loc='lower right')
    plt.show()

def evaluate_metrics(y_true, y_pred, y_pred_probs, num_classes, top_k=5, labels=None):
    metrics = {
        'accuracy': accuracy(y_true, y_pred),
        'precision': precision(y_true, y_pred),
        'recall': recall(y_true, y_pred),
        'f1': f1(y_true, y_pred),
        'top_k_accuracy': top_k_accuracy(y_true, y_pred_probs, k=top_k),
        'mean_class_accuracy': mean_class_accuracy(y_true, y_pred),
        'confusion_matrix': confusion_matrix(y_true, y_pred, num_classes),
        'top_n_error_rate': top_n_error_rate(y_true, y_pred_probs, n=top_k),
        'roc_auc': roc_auc_multiclass(y_true, y_pred_probs, num_classes)
    }
    
    # Plot confusion matrix and ROC curves
    plot_confusion_matrix(y_true, y_pred, num_classes, labels=labels)
    plot_roc_curves(y_true, y_pred_probs, num_classes, labels=labels)
    
    # Class-wise performance
    plot_class_performance(y_true, y_pred, num_classes, labels=labels)
    
    return metrics

# Usage
# y_true = np.array([...])  # Ground truth labels
# y_pred = np.array([...])  # Predicted labels
# y_pred_probs = np.array([...])  # Predicted probabilities
# labels = ['class_0', 'class_1', ..., 'class_n']  # Class names
# metrics = evaluate_metrics(y_true, y_pred, y_pred_probs, num_classes=len(labels), top_k=5, labels=labels)