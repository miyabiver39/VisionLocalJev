import time
import os
import psutil
import threading
from typing import Dict, Any

class MetricsTracker:
    """Tracks system performance, hardware resources, inference latencies, and streaming statistics."""

    def __init__(self):
        self.lock = threading.Lock()
        self.process = psutil.Process(os.getpid())
        
        self.active_ws_clients = 0
        self.total_frames_sampled = 0
        self.total_frames_dropped = 0
        self.total_inferences = 0
        self.total_alerts = 0
        
        self.latest_vision_latency_ms = 0.0
        self.latest_decision_latency_ms = 0.0
        self.latest_rag_latency_ms = 0.0
        
        # Cache CPU percentage readings to avoid blocking
        self.cached_cpu_percent = 0.0
        self.last_cpu_check = 0.0

    def update_inference_telemetry(self, vision_ms: float, decision_ms: float, rag_ms: float = 0.0, is_alert: bool = False):
        """Updates latency metrics for a completed inference cycle."""
        with self.lock:
            self.latest_vision_latency_ms = round(vision_ms, 2)
            self.latest_decision_latency_ms = round(decision_ms, 2)
            self.latest_rag_latency_ms = round(rag_ms, 2)
            self.total_inferences += 1
            if is_alert:
                self.total_alerts += 1

    def record_frame_sampled(self):
        with self.lock:
            self.total_frames_sampled += 1

    def record_frame_dropped(self):
        with self.lock:
            self.total_frames_dropped += 1

    def set_active_connections(self, count: int):
        with self.lock:
            self.active_ws_clients = count

    def get_system_metrics(self) -> Dict[str, Any]:
        """Returns comprehensive resource and telemetry statistics."""
        now = time.time()
        
        # Avoid hammering cpu_percent more than once every 500ms
        if now - self.last_cpu_check >= 0.5:
            self.cached_cpu_percent = psutil.cpu_percent(interval=None)
            self.last_cpu_check = now

        mem = psutil.virtual_memory()
        proc_mem = self.process.memory_info()

        with self.lock:
            return {
                "timestamp": now,
                "system": {
                    "cpu_percent": round(self.cached_cpu_percent, 1),
                    "memory_percent": round(mem.percent, 1),
                    "memory_used_mb": round(mem.used / (1024 * 1024), 1),
                    "memory_total_mb": round(mem.total / (1024 * 1024), 1),
                    "process_memory_mb": round(proc_mem.rss / (1024 * 1024), 1)
                },
                "connections": {
                    "websocket_clients": self.active_ws_clients
                },
                "inference": {
                    "total_cycles": self.total_inferences,
                    "total_alerts": self.total_alerts,
                    "sampled_frames": self.total_frames_sampled,
                    "dropped_frames": self.total_frames_dropped,
                    "vision_latency_ms": self.latest_vision_latency_ms,
                    "decision_latency_ms": self.latest_decision_latency_ms,
                    "rag_latency_ms": self.latest_rag_latency_ms,
                    "total_latency_ms": round(self.latest_vision_latency_ms + self.latest_decision_latency_ms + self.latest_rag_latency_ms, 2)
                }
            }


# Global instance
metrics_tracker = MetricsTracker()
