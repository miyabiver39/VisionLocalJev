import time
import json
import logging
import threading
import requests
from concurrent.futures import ThreadPoolExecutor
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

logger = logging.getLogger("vision_jev.webhook")

MAX_HISTORY = 100


class WebhookConfig(BaseModel):
    id: str
    name: str
    url: str
    format: str = "generic_json"  # slack | discord | generic_json
    enabled: bool = True
    min_score: float = 0.75
    cooldown_seconds: float = 30.0


class WebhookDispatcher:
    """Manages and dispatches real-time incident alert webhooks to Slack, Discord, or Custom endpoints."""

    def __init__(self):
        self.webhooks: Dict[str, WebhookConfig] = {}
        self.lock = threading.Lock()
        self.last_dispatched: Dict[str, float] = {}  # key -> timestamp
        self.history: List[Dict[str, Any]] = []  # max MAX_HISTORY items
        # Bounded worker pool: slow/unreachable endpoints can no longer spawn unbounded threads
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="webhook")

    def add_or_update(self, config: WebhookConfig):
        with self.lock:
            self.webhooks[config.id] = config
            logger.info(f"Webhook '{config.name}' ({config.id}) registered/updated: {config.url}")

    def remove(self, webhook_id: str) -> bool:
        with self.lock:
            if webhook_id in self.webhooks:
                del self.webhooks[webhook_id]
                prefix = f"{webhook_id}:"
                for key in [k for k in self.last_dispatched if k.startswith(prefix)]:
                    del self.last_dispatched[key]
                return True
            return False

    def list_configs(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [w.model_dump() for w in self.webhooks.values()]

    def get_history(self) -> List[Dict[str, Any]]:
        with self.lock:
            return list(reversed(self.history))

    def dispatch_alert_async(self, alert_event: Dict[str, Any]):
        """Selects due webhooks (atomically w.r.t. cooldown) and sends them on the worker pool."""
        for hook in self._select_due_hooks(alert_event):
            self.executor.submit(self._send_payload, hook, alert_event)

    def _select_due_hooks(self, event: Dict[str, Any]) -> List[WebhookConfig]:
        """Returns enabled hooks whose score filter and per-(hook, camera) cooldown allow sending.

        The cooldown check-and-set happens under the lock so concurrent alerts
        cannot both pass the cooldown and send duplicates.
        """
        camera_id = event.get("camera_id", "default")
        score = event.get("score", 0.0)
        now = time.time()
        due: List[WebhookConfig] = []

        with self.lock:
            for hook in self.webhooks.values():
                if not hook.enabled or score < hook.min_score:
                    continue
                cooldown_key = f"{hook.id}:{camera_id}"
                if now - self.last_dispatched.get(cooldown_key, 0.0) < hook.cooldown_seconds:
                    logger.debug(f"Suppressing webhook '{hook.name}' due to cooldown.")
                    continue
                self.last_dispatched[cooldown_key] = now
                due.append(hook)
        return due

    def send_test_ping(self, webhook_id: str) -> Dict[str, Any]:
        """Sends an immediate test alert to verify endpoint connectivity."""
        with self.lock:
            hook = self.webhooks.get(webhook_id)
        if not hook:
            return {"success": False, "error": f"Webhook '{webhook_id}' not found"}

        test_event = {
            "camera_id": "test_cam",
            "camera_name": "Test Surveillance Cam",
            "state": "TEST PING: System verification event initiated by operator.",
            "is_alert": True,
            "alert_reason": "Manual operator connectivity test.",
            "score": 0.99,
            "confidence": 1.0,
            "timestamp": time.time(),
            "time_str": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "sop_action": {
                "title": "System Operational Test",
                "priority": "LOW",
                "procedure_steps": ["Verify network path", "Confirm notification delivery"]
            }
        }
        return self._send_payload(hook, test_event)

    def _send_payload(self, hook: WebhookConfig, event: Dict[str, Any]) -> Dict[str, Any]:
        """Constructs appropriate format and posts to the external URL."""
        payload = {}
        headers = {"Content-Type": "application/json"}

        cam_name = event.get("camera_name", event.get("camera_id", "Unknown"))
        reason = event.get("alert_reason", "Threshold exceeded")
        state_text = event.get("state", "N/A")
        time_str = event.get("time_str", time.strftime("%H:%M:%S", time.localtime()))
        sop = event.get("sop_action", {})

        if hook.format == "slack":
            payload = {
                "text": f"🚨 *[Vision-Jev Guard Alert]* {cam_name} detected incident: {reason}",
                "blocks": [
                    {
                        "type": "header",
                        "text": {"type": "plain_text", "text": "🚨 Vision-Jev Guard Incident Alert", "emoji": True}
                    },
                    {
                        "type": "section",
                        "fields": [
                            {"type": "mrkdwn", "text": f"*Camera:*\n{cam_name}"},
                            {"type": "mrkdwn", "text": f"*Timestamp:*\n{time_str}"},
                            {"type": "mrkdwn", "text": f"*Reason:*\n{reason}"},
                            {"type": "mrkdwn", "text": f"*Confidence/Score:*\n{event.get('score', 0.0):.2f}"}
                        ]
                    },
                    {
                        "type": "section",
                        "text": {"type": "mrkdwn", "text": f"*Detected State:*\n_{state_text}_"}
                    }
                ]
            }
            if sop and sop.get("title"):
                payload["blocks"].append({
                    "type": "context",
                    "elements": [{"type": "mrkdwn", "text": f"*Recommended SOP:* {sop.get('title')} (Priority: {sop.get('priority')})"}]
                })

        elif hook.format == "discord":
            payload = {
                "content": f"🚨 **Vision-Jev Alert Triggered** on `{cam_name}`",
                "embeds": [{
                    "title": f"Incident: {reason}",
                    "color": 15158332,  # Red
                    "fields": [
                        {"name": "Camera", "value": cam_name, "inline": True},
                        {"name": "Time", "value": time_str, "inline": True},
                        {"name": "Detected State", "value": state_text, "inline": False},
                        {"name": "Recommended Action", "value": sop.get("title", "Check camera stream"), "inline": False}
                    ]
                }]
            }
        else:
            # Generic JSON
            payload = {
                "source": "vision-jev-guard",
                "event_type": "security_alert",
                "timestamp": event.get("timestamp", time.time()),
                "camera_id": event.get("camera_id"),
                "camera_name": cam_name,
                "reason": reason,
                "state": state_text,
                "score": event.get("score"),
                "confidence": event.get("confidence"),
                "sop_action": sop
            }

        status_entry = {
            "timestamp": time.time(),
            "time_str": time_str,
            "webhook_id": hook.id,
            "webhook_name": hook.name,
            "url": hook.url,
            "format": hook.format,
            "success": False,
            "status_code": 0,
            "error": ""
        }

        try:
            resp = requests.post(hook.url, json=payload, headers=headers, timeout=5.0)
            status_entry["status_code"] = resp.status_code
            status_entry["success"] = (200 <= resp.status_code < 300)
            if not status_entry["success"]:
                status_entry["error"] = resp.text[:200]
                logger.warning(f"Webhook {hook.name} responded with status {resp.status_code}")
            else:
                logger.info(f"Webhook {hook.name} successfully delivered (status {resp.status_code})")
        except Exception as e:
            status_entry["error"] = str(e)
            logger.error(f"Error dispatching webhook {hook.name}: {e}")

        with self.lock:
            self.history.append(status_entry)
            if len(self.history) > MAX_HISTORY:
                self.history = self.history[-MAX_HISTORY:]

        return status_entry


# Global Webhook Dispatcher instance
webhook_dispatcher = WebhookDispatcher()
