import json
import logging
import os
import smtplib
import urllib.request
from email.mime.text import MIMEText

# ---- Email config ----
_SMTP_HOST     = os.getenv("SMTP_HOST") or os.getenv("EMAIL_HOST")
_SMTP_PORT     = int(os.getenv("SMTP_PORT", "587"))
_SMTP_USER     = os.getenv("SMTP_USERNAME") or os.getenv("EMAIL_USER")
_SMTP_PASS     = os.getenv("SMTP_PASSWORD") or os.getenv("EMAIL_PASS")
_ALERT_FROM    = os.getenv("ALERT_FROM", _SMTP_USER or "alerts@localhost")
_ALERT_TO      = os.getenv("ALERT_TO") or os.getenv("EMAIL_RECIPIENT")

# ---- Telegram config ----
_TG_TOKEN      = os.getenv("TELEGRAM_BOT_TOKEN")
_TG_CHAT_ID    = os.getenv("TELEGRAM_CHAT_ID")


def _send_telegram(subject: str, body: str) -> bool:
    if not (_TG_TOKEN and _TG_CHAT_ID):
        return False
    try:
        text = f"*{subject}*\n```\n{body[:3500]}\n```"
        payload = json.dumps({
            "chat_id": _TG_CHAT_ID,
            "text": text,
            "parse_mode": "Markdown",
        }).encode()
        url = f"https://api.telegram.org/bot{_TG_TOKEN}/sendMessage"
        req = urllib.request.Request(
            url, data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=10) as resp:
            ok = resp.status == 200
        if ok:
            logging.debug("[TELEGRAM] Sent: %s", subject)
        return ok
    except Exception as e:
        logging.warning("[TELEGRAM] Failed to send '%s': %s", subject, e)
        return False


def _send_email(subject: str, body: str) -> bool:
    if not (_SMTP_HOST and _ALERT_TO):
        return False
    try:
        msg = MIMEText(body, "plain", "utf-8")
        msg["Subject"] = subject
        msg["From"]    = _ALERT_FROM
        msg["To"]      = _ALERT_TO
        with smtplib.SMTP(_SMTP_HOST, _SMTP_PORT, timeout=10) as srv:
            srv.starttls()
            if _SMTP_USER and _SMTP_PASS:
                srv.login(_SMTP_USER, _SMTP_PASS)
            srv.sendmail(_ALERT_FROM, _ALERT_TO.split(","), msg.as_string())
        logging.info("[EMAIL] Sent: %s", subject)
        return True
    except Exception as e:
        logging.warning("[EMAIL] Failed to send '%s': %s", subject, e)
        return False


def send_alert(subject: str, payload: dict | None = None) -> bool:
    """
    Send alert via Telegram and/or email, falling back to log-only.
    Channels are independent — both can be active simultaneously.
    """
    body = json.dumps(payload, indent=2) if isinstance(payload, dict) else str(payload or "")

    tg_ok    = _send_telegram(subject, body)
    email_ok = _send_email(subject, body)

    if not tg_ok and not email_ok:
        logging.info("[ALERT] %s | %s", subject, body)
        return False

    return True
