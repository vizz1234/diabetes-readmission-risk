import os
import requests

def send_alert(message: str):
    """
    Sends an alert message. If ALERT_WEBHOOK_URL is set in environment variables,
    it POSTs a JSON text message to the webhook. Otherwise, it prints to console.
    """
    webhook = os.environ.get("ALERT_WEBHOOK_URL")
    if webhook:
        try:
            response = requests.post(webhook, json={"text": message}, timeout=10)
            if response.status_code == 200:
                print(f"[ALERT SENT TO WEBHOOK] {message}")
            else:
                print(f"[ALERT WEBHOOK ERROR {response.status_code}] {message}")
        except Exception as e:
            print(f"[ALERT WEBHOOK FAILED: {e}] {message}")
    else:
        print(f"\n========================================================")
        print(f"[ALERT] {message}")
        print(f"========================================================\n")
