import os
import asyncio
import logging
import json
import time
from collections import deque
from typing import Set, Optional, Dict, Any, List, Deque
from contextlib import asynccontextmanager

import base64
import cv2
import numpy as np

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request, HTTPException
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

from app.camera import camera_manager
from app.vision import VisionExtractor
from app.decision import DecisionEngine
from app.webhook import webhook_dispatcher, WebhookConfig
from app.rag import rag_engine, SOPDocument
from app.visual_rag import visual_rag_engine
from app.metrics import metrics_tracker
from app.security import BasicAuthMiddleware, load_credentials, validate_http_url

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("vision_jev.main")

# Load Environment Variables
CAMERA_SOURCE = os.getenv("CAMERA_SOURCE", "synthetic")
VISION_MODE = os.getenv("VISION_MODE", "detector")
DECISION_ENGINE = os.getenv("DECISION_ENGINE", "diffusion-gemma-jev")
DJEV_MODE = os.getenv("DJEV_MODE", "embedded")
DJEV_SERVER_URL = os.getenv("DJEV_SERVER_URL", "http://localhost:8080/v1/djev/decide")
DJEV_DIFFUSION_STEPS = int(os.getenv("DJEV_DIFFUSION_STEPS", "8"))
SAMPLE_FPS = float(os.getenv("SAMPLE_FPS", "1.0"))
ALERT_THRESHOLD = float(os.getenv("ALERT_THRESHOLD", "0.80"))
RAG_SERVER_URL = os.getenv("RAG_SERVER_URL", "")
MAX_UPLOAD_IMAGE_BYTES = int(os.getenv("MAX_UPLOAD_IMAGE_BYTES", str(10 * 1024 * 1024)))
AUTH_CREDENTIALS = load_credentials()

# Global Components
vision_extractor: Optional[VisionExtractor] = None
decision_engine: Optional[DecisionEngine] = None
active_connections: Set[WebSocket] = set()
pipeline_task: Optional[asyncio.Task] = None

# Recent alert event log (in-memory, up to 100 entries)
MAX_EVENT_LOG = 100
recent_event_log: Deque[Dict[str, Any]] = deque(maxlen=MAX_EVENT_LOG)

# Manual scenario freeze / override state: camera_id -> {state, preset_id, expires_at}
scenario_overrides: Dict[str, Dict[str, Any]] = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager to initialize components, default camera, and background pipeline."""
    global vision_extractor, decision_engine, pipeline_task

    logger.info("Initializing Vision-Jev Guard Platform powered by DiffusionGemma-Jev (DJev)...")

    vision_extractor = VisionExtractor(mode=VISION_MODE)
    decision_engine = DecisionEngine(
        mode=DJEV_MODE,
        remote_url=DJEV_SERVER_URL,
        presets_dir="app/presets",
        alert_threshold=ALERT_THRESHOLD,
        diffusion_steps=DJEV_DIFFUSION_STEPS
    )

    # Register default primary camera from environment
    source_type = "webcam"
    if CAMERA_SOURCE.startswith("rtsp://"):
        source_type = "rtsp"
    elif CAMERA_SOURCE.startswith("http://") or CAMERA_SOURCE.startswith("https://"):
        source_type = "jpeg_url"
    elif CAMERA_SOURCE == "synthetic":
        source_type = "synthetic"

    camera_manager.add_camera(
        camera_id="cam_main",
        name="Primary Camera",
        source_type=source_type,
        source_url=CAMERA_SOURCE,
        preset_id="security",
        sample_fps=SAMPLE_FPS
    )

    # Start the continuous inference loop
    pipeline_task = asyncio.create_task(multi_camera_inference_loop())

    logger.info("Vision-Jev Guard Platform is operational.")
    yield

    # Teardown
    logger.info("Shutting down Vision-Jev Guard...")
    if pipeline_task:
        pipeline_task.cancel()
        try:
            await pipeline_task
        except asyncio.CancelledError:
            pass

    camera_manager.stop_all()
    logger.info("Shutdown complete.")


app = FastAPI(title="Vision-Jev Guard Platform", version="0.2.0", lifespan=lifespan)

# Optional HTTP Basic auth for all HTTP + WebSocket routes (AUTH_USERNAME / AUTH_PASSWORD)
app.add_middleware(BasicAuthMiddleware, credentials=AUTH_CREDENTIALS)
if AUTH_CREDENTIALS is None:
    logger.warning(
        "AUTH_USERNAME / AUTH_PASSWORD are not set: the dashboard, camera feeds and APIs are "
        "accessible without authentication. Set them before exposing this server on a network."
    )

# Mount static files and templates
app.mount("/static", StaticFiles(directory="app/static"), name="static")
templates = Jinja2Templates(directory="app/templates")


# ===================== Request Models =====================

class ThresholdUpdateRequest(BaseModel):
    threshold: float


class CameraCreateRequest(BaseModel):
    camera_id: str
    name: str
    source_type: str  # rtsp, jpeg_url, webcam, synthetic
    source_url: str
    preset_id: str = "security"
    sample_fps: float = 1.0


class CameraUpdateRequest(BaseModel):
    name: Optional[str] = None
    preset_id: Optional[str] = None
    sample_fps: Optional[float] = None


class PresetSelectRequest(BaseModel):
    preset_id: str
    camera_id: Optional[str] = None


class ManualTriggerRequest(BaseModel):
    state: str
    preset_id: Optional[str] = "security"
    camera_id: Optional[str] = "cam_main"
    freeze_seconds: Optional[int] = 20


class RAGQueryRequest(BaseModel):
    query: str
    category: Optional[str] = None


class VisualRAGRegisterRequest(BaseModel):
    title: str
    category: str
    is_anomaly: bool = False
    image_base64: str  # Base64 data URL or raw Base64 JPEG/PNG
    sop_id: Optional[str] = ""
    description: Optional[str] = ""


# ===================== Validation Helpers =====================

def is_valid_preset(preset_id: Optional[str]) -> bool:
    return bool(preset_id) and decision_engine is not None and preset_id in decision_engine.presets


def require_valid_preset(preset_id: Optional[str]):
    """Raises HTTP 400 if the preset id is unknown to the decision engine."""
    if not is_valid_preset(preset_id):
        available = sorted(decision_engine.presets.keys()) if decision_engine else []
        raise HTTPException(status_code=400, detail=f"Unknown preset_id '{preset_id}'. Available: {available}")


# ===================== WebSocket & Broadcast =====================

async def broadcast_ws(message: dict):
    """Broadcasts a JSON message to all connected WebSocket clients."""
    if not active_connections:
        return
    msg_text = json.dumps(message)
    dead_connections = set()
    for ws in list(active_connections):
        try:
            await ws.send_text(msg_text)
        except Exception:
            dead_connections.add(ws)
    for dead in dead_connections:
        active_connections.discard(dead)
    metrics_tracker.set_active_connections(len(active_connections))


# ===================== Background Inference Pipeline =====================

async def process_sample(cam_id: str, frame: np.ndarray, preset_id: str, has_motion: bool) -> Optional[Dict[str, Any]]:
    """Runs one camera sample through Vision -> DJev -> SOP RAG -> Visual RAG -> Webhook -> WS broadcast.

    Returns the composite result broadcast to the WebUI, or None if the sample was skipped.
    """
    cam = camera_manager.get_camera(cam_id)
    cam_name = cam.name if cam else cam_id

    loop = asyncio.get_running_loop()

    # Check if this camera has an active manual test scenario override
    now = time.time()
    active_override = scenario_overrides.get(cam_id)
    is_scenario_override = False
    remaining_override_sec = 0

    if active_override:
        if now < active_override.get("expires_at", 0):
            preset_id = active_override.get("preset_id", preset_id)
            is_scenario_override = True
            remaining_override_sec = max(1, int(active_override["expires_at"] - now))
        else:
            scenario_overrides.pop(cam_id, None)

    # 1. Vision State Extraction
    t0 = time.time()
    vision_res = await loop.run_in_executor(
        None, vision_extractor.extract_state, frame, preset_id, cam_id
    )
    state_text = active_override["state"] if is_scenario_override else vision_res["state"]
    if is_scenario_override:
        has_motion = True
    vision_latency = vision_res["latency_ms"]

    # 2. Decision Engine Evaluation (DiffusionGemma-Jev Multimodal Evaluation)
    decision_res = await loop.run_in_executor(
        None, decision_engine.evaluate_multimodal, frame, preset_id, cam_id, cam_name, state_text
    )
    if "error" in decision_res:
        # Misconfigured preset: skip this sample instead of stalling the whole loop
        logger.warning(f"Skipping inference for camera '{cam_id}': {decision_res['error']}")
        return None
    decision_latency = decision_res["latency_ms"]

    # 3. RAG Knowledge Retrieval (Standard Operating Procedure)
    rag_res = await loop.run_in_executor(
        None, rag_engine.search_sop, state_text, preset_id
    )
    rag_latency = rag_res.get("latency_ms", 0.0)
    sop_action = rag_res.get("sop")

    # 3.5 Visual Example RAG (Image-based cosine reference matching)
    visual_rag_res = await loop.run_in_executor(
        None, visual_rag_engine.match_frame, frame, preset_id
    )
    # If Visual RAG found a high-confidence anomaly reference with linked SOP, prioritize it
    if visual_rag_res.get("is_anomalous") and (visual_rag_res.get("top_match") or {}).get("sop_id"):
        linked_sop_id = visual_rag_res["top_match"]["sop_id"]
        linked_sop = rag_engine.documents.get(linked_sop_id)
        if linked_sop:
            sop_action = linked_sop.model_dump()
            rag_res["matched"] = True
            rag_res["source"] = "visual_example_rag"
            rag_res["relevance_score"] = visual_rag_res["similarity"]

    # 4. Metrics & Telemetry Update
    visual_anomalous = bool(visual_rag_res.get("is_anomalous", False))
    is_alert = decision_res["is_alert"] or visual_anomalous
    alert_score = decision_res["score"]
    alert_reason = ""
    if decision_res["is_alert"]:
        alert_reason = decision_res["alert_reason"]
    elif visual_anomalous:
        top_title = (visual_rag_res.get("top_match") or {}).get("title", "")
        alert_reason = f"Visual Anomaly: {top_title}" if top_title else "Visual Anomaly detected"
        # Visual-only alerts carry the anomaly score so webhook min_score filters don't drop them
        alert_score = max(alert_score, float(visual_rag_res.get("anomaly_score", 0.0)))
    metrics_tracker.update_inference_telemetry(
        vision_latency, decision_latency, rag_latency, is_alert
    )

    # 5. Composite Event Assembly
    time_now = time.time()
    time_str = time.strftime("%H:%M:%S", time.localtime(time_now))
    composite_result = {
        "type": "decision_update",
        "timestamp": time_now,
        "time_str": time_str,
        "camera_id": cam_id,
        "camera_name": cam_name,
        "preset_id": preset_id,
        "preset_name": decision_res.get("preset_name", preset_id),
        "state": state_text,
        "has_motion": has_motion,
        "is_scenario_override": is_scenario_override,
        "remaining_override_sec": remaining_override_sec,
        "vision": {
            "latency_ms": vision_latency,
            "mode": vision_res["mode"]
        },
        "decision": {
            "latency_ms": decision_latency,
            "engine": "diffusion-gemma-jev",
            "decisions": decision_res["decisions"],
            "score": decision_res["score"],
            "is_alert": is_alert,
            "alert_reason": alert_reason,
            "alert_threshold": decision_res["alert_threshold"],
            "djev": decision_res.get("djev", {})
        },
        "visual_rag": {
            "top_match": visual_rag_res.get("top_match"),
            "similarity": visual_rag_res.get("similarity", 0.0),
            "anomaly_score": visual_rag_res.get("anomaly_score", 0.0),
            "is_anomalous": visual_rag_res.get("is_anomalous", False),
            "latency_ms": visual_rag_res.get("latency_ms", 0.0)
        },
        "rag": {
            "source": rag_res.get("source", "embedded_rag"),
            "matched": rag_res.get("matched", False),
            "relevance_score": rag_res.get("relevance_score", 0.0),
            "latency_ms": rag_latency,
            "sop": sop_action
        },
        "metrics": metrics_tracker.get_system_metrics()
    }

    # 6. If Alert Triggered, Dispatch WebHook and Record History
    if is_alert:
        alert_payload = {
            "camera_id": cam_id,
            "camera_name": cam_name,
            "state": state_text,
            "score": alert_score,
            "confidence": alert_score,
            "alert_reason": alert_reason,
            "timestamp": time_now,
            "time_str": time_str,
            "sop_action": sop_action
        }
        webhook_dispatcher.dispatch_alert_async(alert_payload)

        # Record in-memory event log (bounded by deque maxlen)
        recent_event_log.append(alert_payload)

    # 7. Broadcast Telemetry to connected WebUI clients
    await broadcast_ws(composite_result)
    return composite_result


async def multi_camera_inference_loop():
    """
    Asynchronous continuous loop scheduling inference across N cameras with zero-latency drop policy.
    Evaluates: Vision State -> Jev Typed Decision -> RAG SOP Action -> WebHook Dispatch -> WS Broadcast.
    """
    logger.info("Multi-camera inference pipeline loop started.")

    while True:
        try:
            await asyncio.sleep(0.04)  # 25Hz poll tick
            if vision_extractor is None or decision_engine is None:
                continue

            sample = camera_manager.get_next_sample()
            if sample is None:
                continue

            cam_id, frame, preset_id, has_motion = sample
            metrics_tracker.record_frame_sampled()

            await process_sample(cam_id, frame, preset_id, has_motion)

        except asyncio.CancelledError:
            break
        except Exception as e:
            logger.error(f"Error in multi-camera pipeline loop: {e}", exc_info=True)
            await asyncio.sleep(1.0)


# ===================== Health Check =====================

@app.get("/healthz")
async def healthz():
    """Liveness probe (exempt from Basic auth so container health checks work)."""
    return {"status": "ok"}


# ===================== UI Dashboard Route =====================

@app.get("/", response_class=HTMLResponse)
async def index_page(request: Request):
    """Renders the comprehensive surveillance and control dashboard."""
    presets = decision_engine.get_preset_list() if decision_engine else []
    cameras = camera_manager.get_all_status()
    webhooks = webhook_dispatcher.list_configs()
    rag_docs = rag_engine.list_documents()

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={
            "presets": presets,
            "cameras": cameras,
            "webhooks": webhooks,
            "rag_docs": rag_docs,
            "camera_source": CAMERA_SOURCE,
            "vision_mode": VISION_MODE,
            "decision_engine": DECISION_ENGINE,
            "djev_mode": DJEV_MODE,
            "djev_steps": DJEV_DIFFUSION_STEPS,
            "djev_server_url": DJEV_SERVER_URL,
            "alert_threshold": ALERT_THRESHOLD,
            "rag_server_url": RAG_SERVER_URL
        }
    )


# ===================== Video Streaming Routes =====================

async def mjpeg_generator(camera_id: str, request: Optional[Request] = None):
    """Async MJPEG stream frame generator for a specific camera.

    Runs on the event loop (no worker thread is held per viewer) and terminates
    when the camera is removed or the client disconnects.
    """
    last_sent: Optional[bytes] = None
    while True:
        if request is not None and await request.is_disconnected():
            break
        cam = camera_manager.get_camera(camera_id)
        if cam is None:
            break
        jpeg_bytes = cam.get_latest_jpeg()
        if jpeg_bytes and jpeg_bytes is not last_sent:
            last_sent = jpeg_bytes
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg_bytes + b"\r\n"
            )
        await asyncio.sleep(0.04)  # ~25 FPS stream rate


@app.get("/api/cameras/{cam_id}/feed")
async def camera_feed(cam_id: str, request: Request):
    """Streams MJPEG video feed for the specified camera."""
    cam = camera_manager.get_camera(cam_id)
    if not cam:
        raise HTTPException(status_code=404, detail=f"Camera '{cam_id}' not found")
    return StreamingResponse(
        mjpeg_generator(cam_id, request),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/video_feed")
async def default_video_feed(request: Request):
    """Default fallback video feed routing to the first available camera."""
    cameras = camera_manager.get_all_cameras()
    if not cameras:
        raise HTTPException(status_code=404, detail="No camera registered")
    return StreamingResponse(
        mjpeg_generator(cameras[0].camera_id, request),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ===================== WebSocket Route =====================

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """Real-time WebSocket endpoint for telemetry, metrics, decision alerts, and interactive control."""
    await websocket.accept()
    active_connections.add(websocket)
    metrics_tracker.set_active_connections(len(active_connections))
    logger.info(f"WebSocket client connected. Total clients: {len(active_connections)}")

    # Send initial status payload
    initial_payload = {
        "type": "initial_state",
        "cameras": camera_manager.get_all_status(),
        "metrics": metrics_tracker.get_system_metrics()
    }
    await websocket.send_text(json.dumps(initial_payload))

    try:
        while True:
            data_text = await websocket.receive_text()
            try:
                msg = json.loads(data_text)
                action = msg.get("action")
                if action == "select_preset":
                    preset_id = msg.get("preset_id")
                    if not is_valid_preset(preset_id):
                        continue
                    cam_id = msg.get("camera_id")
                    if cam_id:
                        camera_manager.update_camera_preset(cam_id, preset_id)
                    else:
                        for c in camera_manager.get_all_cameras():
                            camera_manager.update_camera_preset(c.camera_id, preset_id)
                    await broadcast_ws({
                        "type": "cameras_updated",
                        "cameras": camera_manager.get_all_status()
                    })
            except json.JSONDecodeError:
                pass
    except WebSocketDisconnect:
        active_connections.discard(websocket)
        metrics_tracker.set_active_connections(len(active_connections))
        logger.info(f"WebSocket client disconnected. Total clients: {len(active_connections)}")
    except Exception as e:
        active_connections.discard(websocket)
        metrics_tracker.set_active_connections(len(active_connections))
        logger.warning(f"WebSocket error: {e}")


@app.post("/api/settings/threshold")
async def update_threshold(req: ThresholdUpdateRequest):
    """Dynamically updates the global safety alert threshold."""
    global ALERT_THRESHOLD
    ALERT_THRESHOLD = max(0.1, min(0.99, req.threshold))
    if decision_engine:
        decision_engine.alert_threshold = ALERT_THRESHOLD
    logger.info(f"Updated global alert threshold to: {ALERT_THRESHOLD}")
    await broadcast_ws({
        "type": "threshold_updated",
        "alert_threshold": ALERT_THRESHOLD
    })
    return JSONResponse({"success": True, "alert_threshold": ALERT_THRESHOLD})


# ===================== Camera Management REST API =====================

@app.get("/api/cameras")
async def list_cameras():
    """Returns list and real-time status of all active cameras."""
    return JSONResponse(camera_manager.get_all_status())


@app.post("/api/cameras")
async def add_camera(req: CameraCreateRequest):
    """Registers and starts a new camera (RTSP, JPEG URL, Webcam, or Synthetic)."""
    require_valid_preset(req.preset_id)
    if req.source_type.lower() == "jpeg_url":
        try:
            validate_http_url(req.source_url)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
    try:
        cam = camera_manager.add_camera(
            camera_id=req.camera_id,
            name=req.name,
            source_type=req.source_type,
            source_url=req.source_url,
            preset_id=req.preset_id,
            sample_fps=req.sample_fps
        )
        await broadcast_ws({
            "type": "cameras_updated",
            "cameras": camera_manager.get_all_status()
        })
        return JSONResponse({"success": True, "camera": cam.get_status()})
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/cameras/{cam_id}")
async def get_camera(cam_id: str):
    """Fetches metadata and status for a single camera."""
    cam = camera_manager.get_camera(cam_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")
    return JSONResponse(cam.get_status())


@app.put("/api/cameras/{cam_id}")
async def update_camera(cam_id: str, req: CameraUpdateRequest):
    """Updates an existing camera configuration."""
    cam = camera_manager.get_camera(cam_id)
    if not cam:
        raise HTTPException(status_code=404, detail="Camera not found")

    if req.preset_id is not None:
        require_valid_preset(req.preset_id)
    if req.sample_fps is not None and req.sample_fps <= 0:
        raise HTTPException(status_code=400, detail="sample_fps must be greater than 0")

    if req.name is not None:
        cam.name = req.name
    if req.preset_id is not None:
        cam.preset_id = req.preset_id
    if req.sample_fps is not None:
        cam.set_sample_fps(req.sample_fps)

    await broadcast_ws({
        "type": "cameras_updated",
        "cameras": camera_manager.get_all_status()
    })
    return JSONResponse({"success": True, "camera": cam.get_status()})


@app.delete("/api/cameras/{cam_id}")
async def delete_camera(cam_id: str):
    """Stops and removes a camera from the system."""
    success = camera_manager.remove_camera(cam_id)
    if not success:
        raise HTTPException(status_code=404, detail="Camera not found")
    if vision_extractor:
        vision_extractor.reset_camera(cam_id)

    await broadcast_ws({
        "type": "cameras_updated",
        "cameras": camera_manager.get_all_status()
    })
    return JSONResponse({"success": True, "deleted": cam_id})


# ===================== WebHook Management REST API =====================

@app.get("/api/webhooks")
async def list_webhooks():
    return JSONResponse(webhook_dispatcher.list_configs())


@app.post("/api/webhooks")
async def save_webhook(config: WebhookConfig):
    webhook_dispatcher.add_or_update(config)
    return JSONResponse({"success": True, "webhook": config.model_dump()})


@app.delete("/api/webhooks/{webhook_id}")
async def delete_webhook(webhook_id: str):
    success = webhook_dispatcher.remove(webhook_id)
    if not success:
        raise HTTPException(status_code=404, detail="Webhook not found")
    return JSONResponse({"success": True, "deleted": webhook_id})


@app.post("/api/webhooks/{webhook_id}/test")
async def test_webhook(webhook_id: str):
    """Sends an immediate verification ping to the specified webhook."""
    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, webhook_dispatcher.send_test_ping, webhook_id)
    return JSONResponse(result)


@app.get("/api/webhooks/history")
async def get_webhook_history():
    return JSONResponse(webhook_dispatcher.get_history())


# ===================== RAG Knowledge REST API =====================

@app.get("/api/rag/documents")
async def list_rag_documents():
    """Lists all standard operating procedure (SOP) documents in knowledge base."""
    return JSONResponse(rag_engine.list_documents())


@app.post("/api/rag/documents")
async def add_rag_document(doc: SOPDocument):
    """Appends or updates an SOP document in the knowledge base."""
    rag_engine.add_document(doc)
    return JSONResponse({"success": True, "document": doc.model_dump()})


@app.post("/api/rag/search")
async def query_rag(req: RAGQueryRequest):
    """Executes SOP retrieval against a situation description."""
    res = rag_engine.search_sop(req.query, req.category)
    return JSONResponse(res)


# ===================== Telemetry, Presets & Scenarios =====================

@app.get("/api/metrics")
async def get_system_metrics():
    """Returns detailed real-time performance and resource telemetry."""
    return JSONResponse(metrics_tracker.get_system_metrics())


@app.get("/api/events")
async def get_recent_events():
    """Returns recently recorded incident alerts."""
    return JSONResponse(list(reversed(recent_event_log)))


@app.get("/api/presets")
async def get_presets():
    if not decision_engine:
        return JSONResponse([])
    return JSONResponse(decision_engine.get_preset_list())


@app.post("/api/trigger_scenario")
async def trigger_manual_scenario(req: ManualTriggerRequest):
    """Manually injects situation text for instant deterministic verification."""
    if not decision_engine:
        raise HTTPException(status_code=500, detail="Decision engine not initialized")

    cam = camera_manager.get_camera(req.camera_id or "cam_main")
    cam_name = cam.name if cam else (req.camera_id or "Manual Injection")
    preset_id = req.preset_id or "security"
    require_valid_preset(preset_id)

    eval_res = decision_engine.evaluate(req.state, preset_id, req.camera_id or "manual", cam_name)
    rag_res = rag_engine.search_sop(req.state, preset_id)
    eval_res["sop_action"] = rag_res.get("sop")

    if eval_res["is_alert"]:
        alert_event = {
            "camera_id": req.camera_id or "manual",
            "camera_name": cam_name,
            "state": req.state,
            "score": eval_res["score"],
            "confidence": eval_res["score"],
            "alert_reason": eval_res["alert_reason"],
            "timestamp": time.time(),
            "time_str": time.strftime("%H:%M:%S", time.localtime()),
            "sop_action": rag_res.get("sop")
        }
        webhook_dispatcher.dispatch_alert_async(alert_event)
        recent_event_log.append(alert_event)

    # Set temporary scenario freeze on target camera (or cam_main)
    target_cam_id = req.camera_id or "cam_main"
    freeze_sec = req.freeze_seconds if req.freeze_seconds is not None else 20
    if freeze_sec > 0:
        scenario_overrides[target_cam_id] = {
            "state": req.state,
            "preset_id": preset_id,
            "expires_at": time.time() + freeze_sec
        }
        logger.info(f"Manual scenario override active on '{target_cam_id}' for {freeze_sec}s")

    return JSONResponse(eval_res)


@app.post("/api/resume_live")
async def resume_live_monitoring(req: Optional[Dict[str, Any]] = None):
    """Cancels any active manual scenario injection freeze and resumes live surveillance."""
    cam_id = req.get("camera_id") if req else None
    if cam_id:
        scenario_overrides.pop(cam_id, None)
    else:
        scenario_overrides.clear()
    logger.info("Resumed live surveillance monitoring (manual scenario freeze cleared).")
    return JSONResponse({"success": True, "message": "Resumed live surveillance"})


# ===================== Visual Example RAG API =====================

@app.get("/api/visual_rag/references")
async def list_visual_references(category: Optional[str] = None):
    """Returns all registered visual reference images and baselines."""
    refs = visual_rag_engine.get_all_references()
    if category:
        refs = [r for r in refs if r["category"] == category or r["category"] == "default"]
    return JSONResponse(refs)


@app.post("/api/visual_rag/register")
async def register_visual_reference(req: VisualRAGRegisterRequest):
    """Registers a new normal baseline or anomaly case image."""
    try:
        # Decode base64 image
        b64_str = req.image_base64
        if "," in b64_str:
            b64_str = b64_str.split(",", 1)[1]
        # Base64 inflates by 4/3: reject oversized payloads before decoding
        if len(b64_str) * 3 // 4 > MAX_UPLOAD_IMAGE_BYTES:
            raise HTTPException(status_code=413, detail=f"Image exceeds {MAX_UPLOAD_IMAGE_BYTES} bytes")
        img_bytes = base64.b64decode(b64_str)
        img_arr = np.frombuffer(img_bytes, dtype=np.uint8)
        frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image data")

        ref = visual_rag_engine.register_reference(
            title=req.title,
            category=req.category,
            is_anomaly=req.is_anomaly,
            frame=frame,
            sop_id=req.sop_id or "",
            description=req.description or ""
        )
        return JSONResponse({"success": True, "reference": ref.to_dict()})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.delete("/api/visual_rag/references/{ref_id}")
async def delete_visual_reference(ref_id: str):
    """Deletes a visual reference image."""
    deleted = visual_rag_engine.delete_reference(ref_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Reference not found")
    return JSONResponse({"success": True, "ref_id": ref_id})

