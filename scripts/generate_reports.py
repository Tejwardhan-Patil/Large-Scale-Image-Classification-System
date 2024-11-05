import os
import json
import matplotlib.pyplot as plt
import pandas as pd
from utils.metrics import calculate_metrics
from utils.visualization import plot_confusion_matrix, plot_roc_curve, plot_loss_curve

# Paths to directories and files
MODEL_METADATA_PATH = "models/pretrained/model_metadata.json"
EVALUATION_RESULTS_PATH = "results/evaluation_results.json"
REPORTS_DIR = "reports"

# Ensure the reports directory exists
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

# Load model metadata and evaluation results
with open(MODEL_METADATA_PATH, "r") as metadata_file:
    model_metadata = json.load(metadata_file)

with open(EVALUATION_RESULTS_PATH, "r") as results_file:
    evaluation_results = json.load(results_file)

# Generate evaluation metrics summary
metrics = calculate_metrics(evaluation_results['true_labels'], evaluation_results['predicted_labels'])
metrics_summary = pd.DataFrame(metrics, index=[0])

# Save metrics summary as CSV
metrics_summary.to_csv(os.path.join(REPORTS_DIR, "metrics_summary.csv"), index=False)

# Visualize and save confusion matrix
conf_matrix_fig = plot_confusion_matrix(evaluation_results['true_labels'], evaluation_results['predicted_labels'])
conf_matrix_fig.savefig(os.path.join(REPORTS_DIR, "confusion_matrix.png"))

# Visualize and save ROC curve
roc_curve_fig = plot_roc_curve(evaluation_results['true_labels'], evaluation_results['predicted_probs'])
roc_curve_fig.savefig(os.path.join(REPORTS_DIR, "roc_curve.png"))

# Visualize and save loss curve from training
loss_curve_fig = plot_loss_curve(evaluation_results['training_loss'], evaluation_results['validation_loss'])
loss_curve_fig.savefig(os.path.join(REPORTS_DIR, "loss_curve.png"))

# Generate a detailed text report
report_text = f"""
Model Report
============
Model: {model_metadata['model_name']}
Architecture: {model_metadata['architecture']}
Training Date: {model_metadata['training_date']}
Dataset: {model_metadata['dataset_name']}

Evaluation Metrics:
-------------------
Accuracy: {metrics['accuracy']}
Precision: {metrics['precision']}
Recall: {metrics['recall']}
F1-Score: {metrics['f1_score']}
AUC: {metrics['auc']}

Confusion Matrix, ROC Curve, and Loss Curve have been saved to the reports directory.
"""

# Save the report as a text file
with open(os.path.join(REPORTS_DIR, "model_report.txt"), "w") as report_file:
    report_file.write(report_text)

print("Report generation complete. Reports saved to:", REPORTS_DIR)