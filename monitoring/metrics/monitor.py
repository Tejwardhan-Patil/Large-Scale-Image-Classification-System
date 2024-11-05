import time
import logging
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
import psutil
import smtplib
from email.mime.text import MIMEText

# Setting up the logging configuration
logging.basicConfig(filename='model_performance.log', level=logging.INFO, 
                    format='%(asctime)s - %(levelname)s - %(message)s')

# Thresholds for alerting
ALERT_THRESHOLDS = {
    'accuracy': 0.8,
    'f1_score': 0.75,
    'cpu_usage': 80,    # in percentage
    'memory_usage': 80  # in percentage
}

# Email configuration for alerts
EMAIL_CONFIG = {
    'sender': 'monitoring@website.com',
    'recipient': 'admin@website.com',
    'smtp_server': 'smtp.website.com',
    'smtp_port': 587,
    'login': 'monitoring@website.com',
    'password': 'secure_password'
}

# Function to send alerts via email
def send_alert(subject, message):
    """ Sends an alert via email when a threshold is breached. """
    msg = MIMEText(message)
    msg['Subject'] = subject
    msg['From'] = EMAIL_CONFIG['sender']
    msg['To'] = EMAIL_CONFIG['recipient']

    try:
        with smtplib.SMTP(EMAIL_CONFIG['smtp_server'], EMAIL_CONFIG['smtp_port']) as server:
            server.starttls()
            server.login(EMAIL_CONFIG['login'], EMAIL_CONFIG['password'])
            server.sendmail(EMAIL_CONFIG['sender'], EMAIL_CONFIG['recipient'], msg.as_string())
        logging.info("Alert sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send alert: {str(e)}")

# Function to log model metrics
def log_metrics(y_true, y_pred, start_time):
    """ Logs accuracy, F1-score, precision, recall, confusion matrix, and inference time. """
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    precision = precision_score(y_true, y_pred, average='weighted')
    recall = recall_score(y_true, y_pred, average='weighted')
    
    # Compute inference time
    end_time = time.time()
    inference_time = end_time - start_time
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Log the metrics
    logging.info(f"Accuracy: {accuracy:.4f}, F1-score: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, Inference Time: {inference_time:.4f} seconds")
    logging.info(f"Confusion Matrix:\n{cm}")
    
    # Check if any thresholds are breached
    check_alerts(accuracy, f1)

# Function to check if model performance drops below certain thresholds
def check_alerts(accuracy, f1):
    """ Checks if model performance metrics breach thresholds and sends alerts. """
    
    if accuracy < ALERT_THRESHOLDS['accuracy']:
        message = f"Model accuracy dropped below the threshold: {accuracy:.4f}"
        send_alert("Model Accuracy Alert", message)
        logging.warning(message)
    
    if f1 < ALERT_THRESHOLDS['f1_score']:
        message = f"Model F1-score dropped below the threshold: {f1:.4f}"
        send_alert("Model F1-score Alert", message)
        logging.warning(message)

# Function to log system metrics such as CPU, memory, and disk usage
def log_system_metrics():
    """ Logs CPU, memory, disk usage, and network stats. """
    
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()
    disk_usage = psutil.disk_usage('/')
    net_io = psutil.net_io_counters()
    
    # Log system metrics
    logging.info(f"CPU Usage: {cpu_usage}%, Memory Usage: {memory_info.percent}%, Disk Usage: {disk_usage.percent}%")
    logging.info(f"Network - Sent: {net_io.bytes_sent / 1024**2:.2f} MB, Received: {net_io.bytes_recv / 1024**2:.2f} MB")
    
    # Check for alerts on system metrics
    check_system_alerts(cpu_usage, memory_info.percent)

# Function to check if system metrics breach thresholds and send alerts
def check_system_alerts(cpu_usage, memory_usage):
    """ Sends alerts if system resources exceed defined thresholds. """
    
    if cpu_usage > ALERT_THRESHOLDS['cpu_usage']:
        message = f"CPU usage exceeded the threshold: {cpu_usage}%"
        send_alert("CPU Usage Alert", message)
        logging.warning(message)
    
    if memory_usage > ALERT_THRESHOLDS['memory_usage']:
        message = f"Memory usage exceeded the threshold: {memory_usage}%"
        send_alert("Memory Usage Alert", message)
        logging.warning(message)

# Function to detect model drift by comparing recent accuracy/f1 to historical averages
def detect_model_drift(history, current_metric, metric_name):
    """ Detects model drift by comparing the current metric to the historical average. """
    
    avg_metric = sum(history) / len(history)
    
    if current_metric < avg_metric * 0.9:
        message = f"Potential model drift detected: {metric_name} dropped by more than 10% compared to historical average."
        send_alert("Model Drift Alert", message)
        logging.warning(message)

# Main function to track performance, drift, and system metrics
def track_performance(y_true, y_pred, metric_history):
    """ Main function to track and log model and system performance. """
    
    # Record start time
    start_time = time.time()
    
    # Log model performance metrics
    log_metrics(y_true, y_pred, start_time)
    
    # Log system metrics
    log_system_metrics()
    
    # Detect model drift
    accuracy = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, average='weighted')
    
    detect_model_drift(metric_history['accuracy'], accuracy, 'Accuracy')
    detect_model_drift(metric_history['f1'], f1, 'F1-score')

# Historical performance for detecting drift
metric_history = {
    'accuracy': [0.85, 0.84, 0.86, 0.83],
    'f1': [0.78, 0.79, 0.80, 0.77]
}

# Usage
y_true = [0, 1, 1, 0, 1, 0, 0, 1]  # Ground truth labels
y_pred = [0, 1, 0, 0, 1, 0, 1, 1]  # Model predictions

# Track and log performance metrics
track_performance(y_true, y_pred, metric_history)