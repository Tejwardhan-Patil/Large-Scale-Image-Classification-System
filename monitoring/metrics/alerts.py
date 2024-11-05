import smtplib
import requests
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from monitor import get_latest_metrics
from logger import log_error, log_info
import time
import json

# Configurations
ALERT_THRESHOLD = {
    "accuracy": 0.80,
    "f1_score": 0.75,
    "latency": 500  # in milliseconds
}
ALERT_EMAIL = "alerts@website.com"
SMTP_SERVER = "smtp.website.com"
SMTP_PORT = 587
SMTP_USERNAME = "smtp_user"
SMTP_PASSWORD = "smtp_password"
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5  # seconds

# Slack integration
SLACK_WEBHOOK_URL = "https://hooks.slack.com/services/T0000/B0000/XXXXXX"
SLACK_CHANNEL = "#model-alerts"

# PagerDuty integration
PAGERDUTY_API_KEY = "pagerduty_api_key"
PAGERDUTY_SERVICE_ID = "pagerduty_service_id"

# Function to send email alerts
def send_email_alert(subject, body):
    msg = MIMEMultipart()
    msg['From'] = SMTP_USERNAME
    msg['To'] = ALERT_EMAIL
    msg['Subject'] = subject
    msg.attach(MIMEText(body, 'plain'))

    success = False
    for attempt in range(RETRY_ATTEMPTS):
        try:
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
            server.login(SMTP_USERNAME, SMTP_PASSWORD)
            server.sendmail(SMTP_USERNAME, ALERT_EMAIL, msg.as_string())
            server.quit()
            log_info(f"Email alert sent: {subject}")
            success = True
            break
        except Exception as e:
            log_error(f"Failed to send email (Attempt {attempt+1}): {str(e)}")
            time.sleep(RETRY_DELAY)

    if not success:
        log_error("All email alert attempts failed.")

# Function to send Slack alerts
def send_slack_alert(message):
    payload = {
        "channel": SLACK_CHANNEL,
        "text": message
    }
    success = False
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.post(SLACK_WEBHOOK_URL, json=payload)
            if response.status_code == 200:
                log_info("Slack alert sent successfully.")
                success = True
                break
            else:
                log_error(f"Slack alert failed (Attempt {attempt+1}): {response.status_code}")
        except Exception as e:
            log_error(f"Error sending Slack alert (Attempt {attempt+1}): {str(e)}")
        time.sleep(RETRY_DELAY)

    if not success:
        log_error("All Slack alert attempts failed.")

# Function to send PagerDuty alerts
def send_pagerduty_alert(description):
    headers = {
        "Authorization": f"Token token={PAGERDUTY_API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        "incident": {
            "type": "incident",
            "title": "Model Performance Degradation Alert",
            "service": {"id": PAGERDUTY_SERVICE_ID, "type": "service_reference"},
            "body": {"type": "incident_body", "details": description}
        }
    }
    success = False
    for attempt in range(RETRY_ATTEMPTS):
        try:
            response = requests.post("https://api.pagerduty.com/incidents", json=payload, headers=headers)
            if response.status_code == 201:
                log_info("PagerDuty alert triggered.")
                success = True
                break
            else:
                log_error(f"PagerDuty alert failed (Attempt {attempt+1}): {response.status_code}")
        except Exception as e:
            log_error(f"Error triggering PagerDuty (Attempt {attempt+1}): {str(e)}")
        time.sleep(RETRY_DELAY)

    if not success:
        log_error("All PagerDuty alert attempts failed.")

# Function to log metrics
def log_metrics(metrics):
    try:
        with open("metrics_log.json", "a") as f:
            json.dump(metrics, f)
            f.write("\n")
        log_info("Metrics successfully logged.")
    except Exception as e:
        log_error(f"Failed to log metrics: {str(e)}")

# Function to check thresholds and determine if alerts need to be triggered
def check_thresholds(metrics):
    alert_triggered = False
    message = "Model Performance Degradation Detected:\n"

    if metrics["accuracy"] < ALERT_THRESHOLD["accuracy"]:
        message += f"- Accuracy dropped below {ALERT_THRESHOLD['accuracy']}: Current value: {metrics['accuracy']}\n"
        alert_triggered = True

    if metrics["f1_score"] < ALERT_THRESHOLD["f1_score"]:
        message += f"- F1 Score dropped below {ALERT_THRESHOLD['f1_score']}: Current value: {metrics['f1_score']}\n"
        alert_triggered = True

    if metrics["latency"] > ALERT_THRESHOLD["latency"]:
        message += f"- Latency exceeded {ALERT_THRESHOLD['latency']}ms: Current value: {metrics['latency']}ms\n"
        alert_triggered = True

    return alert_triggered, message

# Function to handle multiple alert systems
def trigger_alerts(message):
    send_email_alert("Model Performance Alert", message)
    send_slack_alert(message)
    send_pagerduty_alert(message)

# Function to retrieve and monitor metrics
def check_metrics_and_alert():
    metrics = get_latest_metrics()
    log_metrics(metrics)

    alert_triggered, message = check_thresholds(metrics)

    if alert_triggered:
        log_info("Alert triggered. Sending notifications.")
        trigger_alerts(message)
    else:
        log_info("No alert triggered. Model performance is within thresholds.")

# Retry mechanism for retrieving metrics
def get_metrics_with_retries():
    success = False
    metrics = None
    for attempt in range(RETRY_ATTEMPTS):
        try:
            metrics = get_latest_metrics()
            success = True
            break
        except Exception as e:
            log_error(f"Error retrieving metrics (Attempt {attempt+1}): {str(e)}")
        time.sleep(RETRY_DELAY)

    if not success:
        log_error("All attempts to retrieve metrics failed.")
        return None

    return metrics

# Function to generate a summary report of alerts
def generate_alert_report(alerts):
    try:
        with open("alert_report.txt", "w") as f:
            for alert in alerts:
                f.write(f"Alert: {alert['type']}\n")
                f.write(f"Timestamp: {alert['timestamp']}\n")
                f.write(f"Details: {alert['details']}\n")
                f.write("-" * 40 + "\n")
        log_info("Alert report successfully generated.")
    except Exception as e:
        log_error(f"Failed to generate alert report: {str(e)}")

# Function to aggregate and report alerts
def aggregate_alerts():
    alerts = []
    # Simulating retrieval of past alerts
    past_alerts = [
        {"type": "Accuracy", "timestamp": "2024-09-14 10:00:00", "details": "Accuracy dropped to 0.75"},
        {"type": "F1 Score", "timestamp": "2024-09-13 12:30:00", "details": "F1 Score dropped to 0.70"},
    ]
    for alert in past_alerts:
        alerts.append(alert)

    generate_alert_report(alerts)

if __name__ == "__main__":
    metrics = get_metrics_with_retries()
    if metrics:
        check_metrics_and_alert()
    aggregate_alerts()