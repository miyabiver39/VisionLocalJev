import cv2
import numpy as np
import threading
import time
import os
import logging
import requests
from typing import Optional, Tuple, Dict, Any, List

logger = logging.getLogger("vision_jev.camera")


def resolve_youtube_url(url: str) -> Tuple[Optional[str], str]:
    """Extracts direct streamable URL from YouTube video or live broadcast.
    Supports optional cookies.txt and multiple fallback player client strategies.
    Returns (stream_url, friendly_error_message).
    """
    try:
        import yt_dlp

        # Check for cookies.txt in current directory or env var
        cookie_file = None
        for candidate in ["cookies.txt", "youtube_cookies.txt", os.getenv("YOUTUBE_COOKIE_FILE", "")]:
            if candidate and os.path.exists(candidate):
                cookie_file = candidate
                logger.info(f"Using YouTube cookie file: {cookie_file}")
                break

        # Fallback player client strategies to maximize bypass capability
        strategies = [
            {},  # Standard default extractor
            {'extractor_args': {'youtube': {'player_client': ['mweb', 'android', 'web']}}},
            {'extractor_args': {'youtube': {'player_client': ['ios']}}},
        ]

        last_error = ""
        for strat in strategies:
            ydl_opts = {
                'format': 'bestvideo[height<=720][ext=mp4]/best[height<=720]/best',
                'quiet': True,
                'no_warnings': True,
                **strat
            }
            if cookie_file:
                ydl_opts['cookiefile'] = cookie_file

            try:
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=False)
                    stream_url = info.get('url')
                    if stream_url:
                        logger.info(f"Successfully resolved YouTube stream: '{info.get('title')}'")
                        return stream_url, ""
            except Exception as e:
                last_error = str(e)

        # Generate friendly, actionable error message for the user
        if "Sign in to confirm you" in last_error or "bot" in last_error.lower():
            friendly_err = "YouTube BotGuard遮断: YouTubeが未認証アクセスを制限しています。cookies.txtを配置するか、HLS/直接動画URLをご利用ください。"
        elif "removed" in last_error or "violating" in last_error.lower():
            friendly_err = "YouTube動画が規約違反等で削除されています。"
        elif "unavailable" in last_error.lower():
            friendly_err = "YouTube動画が現在利用不可または非公開です。"
        else:
            friendly_err = f"YouTube解析失敗: {last_error[:90]}"

        logger.warning(f"YouTube resolution notice for '{url}': {friendly_err}")
        return None, friendly_err

    except Exception as e:
        logger.error(f"Failed to resolve YouTube URL '{url}': {e}")
        return None, str(e)


class CameraDevice:
    """Represents a single video stream ingestion worker (RTSP, YouTube, JPEG URL, Webcam, or Synthetic)."""

    def __init__(
        self,
        camera_id: str,
        name: str,
        source_type: str,
        source_url: str,
        preset_id: str = "security",
        sample_fps: float = 1.0,
        motion_threshold: float = 1.5
    ):
        self.camera_id = camera_id
        self.name = name
        self.source_type = source_type.lower()  # rtsp, youtube, jpeg_url, webcam, synthetic
        self.source_url = source_url
        self.preset_id = preset_id
        self.sample_fps = max(0.2, min(sample_fps, 10.0))
        self.sample_interval = 1.0 / self.sample_fps
        self.motion_threshold = motion_threshold

        # Auto-detect source types
        if "youtube.com" in self.source_url or "youtu.be" in self.source_url:
            self.source_type = "youtube"
        elif ".m3u8" in self.source_url:
            self.source_type = "hls"
        elif self.source_url.lower().endswith((".mp4", ".avi", ".mkv", ".mov")):
            self.source_type = "file"

        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()

        self.latest_frame: Optional[np.ndarray] = None
        self.latest_jpeg: Optional[bytes] = None
        self.last_sample_time = 0.0
        self.prev_gray_frame: Optional[np.ndarray] = None

        self.is_synthetic = (self.source_type == "synthetic")
        self.synthetic_angle = 0.0
        self.fps_counter = 0
        self.actual_fps = 0.0
        self.last_fps_calc_time = time.time()
        self.status = "INITIALIZING"
        self.error_message = ""

    def start(self):
        """Starts the ingestion thread."""
        if self.is_running:
            return
        self.is_running = True
        self.status = "CONNECTING"
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()
        logger.info(f"Camera '{self.name}' ({self.camera_id}) worker started.")

    def stop(self):
        """Stops the ingestion thread."""
        self.is_running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.status = "STOPPED"
        logger.info(f"Camera '{self.name}' ({self.camera_id}) stopped.")

    def _open_stream(self) -> bool:
        """Attempts to open video source."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None

        if self.source_type == "synthetic":
            self.is_synthetic = True
            self.status = "ACTIVE"
            return True

        if self.source_type == "jpeg_url":
            self.is_synthetic = False
            self.status = "ACTIVE"
            return True

        target_source = self.source_url

        # YouTube URL handling
        if self.source_type == "youtube":
            logger.info(f"Resolving YouTube stream for {self.name}: {self.source_url}")
            resolved_url, err_msg = resolve_youtube_url(self.source_url)
            if resolved_url:
                target_source = resolved_url
            else:
                self.is_synthetic = True
                self.status = "FALLBACK_SYNTHETIC"
                self.error_message = err_msg or "YouTube stream resolution failed"
                return False

        try:
            if self.source_type == "webcam":
                if str(target_source).isdigit():
                    target_source = int(target_source)
                elif str(target_source).startswith("/dev/video"):
                    try:
                        target_source = int(str(target_source).replace("/dev/video", ""))
                    except ValueError:
                        pass
                self.cap = cv2.VideoCapture(target_source)
            elif self.source_type == "rtsp":
                os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"
                self.cap = cv2.VideoCapture(target_source, cv2.CAP_FFMPEG)
            else:
                self.cap = cv2.VideoCapture(target_source)

            if self.cap is not None and self.cap.isOpened():
                self.is_synthetic = False
                self.status = "ACTIVE"
                self.error_message = ""
                logger.info(f"Camera '{self.name}' opened stream successfully.")
                return True
        except Exception as e:
            self.error_message = str(e)
            logger.warning(f"Camera '{self.name}' failed to open {self.source_url}: {e}")

        self.is_synthetic = True
        self.status = "FALLBACK_SYNTHETIC"
        logger.warning(f"Camera '{self.name}' fell back to synthetic test pattern.")
        return False

    def _fetch_jpeg_url(self) -> Optional[np.ndarray]:
        """Fetches a single JPEG frame via HTTP GET."""
        try:
            resp = requests.get(self.source_url, timeout=1.5)
            if resp.status_code == 200 and resp.content:
                img_arr = np.frombuffer(resp.content, dtype=np.uint8)
                frame = cv2.imdecode(img_arr, cv2.IMREAD_COLOR)
                return frame
        except Exception as e:
            self.error_message = str(e)
        return None

    def _run_loop(self):
        """Main frame acquisition loop with automatic loopback and reconnection."""
        self._open_stream()
        retry_interval = 5.0
        last_retry_time = 0.0

        while self.is_running:
            frame = None

            if self.source_type == "jpeg_url" and not self.is_synthetic:
                frame = self._fetch_jpeg_url()
                time.sleep(0.1)
            elif not self.is_synthetic and self.cap is not None and self.cap.isOpened():
                ret, grabbed = self.cap.read()
                if ret and grabbed is not None:
                    frame = grabbed
                    self.status = "ACTIVE"
                else:
                    # End of file or stream pause -> Rewind video if it's a file/YouTube video
                    if self.source_type in ["youtube", "file"]:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        ret_rewind, grabbed_rewind = self.cap.read()
                        if ret_rewind and grabbed_rewind is not None:
                            frame = grabbed_rewind
                            self.status = "ACTIVE"
                        else:
                            self.status = "RECONNECTING"
                            now = time.time()
                            if now - last_retry_time > retry_interval:
                                last_retry_time = now
                                self._open_stream()
                    else:
                        self.status = "RECONNECTING"
                        time.sleep(0.2)
                        now = time.time()
                        if now - last_retry_time > retry_interval:
                            last_retry_time = now
                            self._open_stream()
            else:
                if not self.is_synthetic:
                    now = time.time()
                    if now - last_retry_time > retry_interval:
                        last_retry_time = now
                        if self._open_stream():
                            continue

            # Fallback to synthetic if no frame available
            if frame is None:
                frame = self._generate_synthetic_frame()
                time.sleep(0.04)

            # Compress JPEG for Web streaming
            ret_encode, jpeg_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

            with self.lock:
                self.latest_frame = frame
                if ret_encode:
                    self.latest_jpeg = jpeg_bytes.tobytes()

            # Measure FPS
            self.fps_counter += 1
            now = time.time()
            if now - self.last_fps_calc_time >= 1.0:
                self.actual_fps = self.fps_counter / (now - self.last_fps_calc_time)
                self.fps_counter = 0
                self.last_fps_calc_time = now

    def _generate_synthetic_frame(self) -> np.ndarray:
        """Generates a test pattern frame (only used when real camera is unavailable)."""
        width, height = 960, 540
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (20, 24, 30)

        # Draw grid
        for y in range(0, height, 40):
            cv2.line(frame, (0, y), (width, y), (30, 35, 45), 1)
        for x in range(0, width, 40):
            cv2.line(frame, (x, 0), (x, height), (30, 35, 45), 1)

        # Draw moving target
        self.synthetic_angle += 0.05
        cx = int(width / 2 + 250 * np.cos(self.synthetic_angle))
        cy = int(height / 2 + 120 * np.sin(self.synthetic_angle * 1.3))
        
        color = (0, 165, 255) if self.preset_id == "security" else (0, 90, 255)
        cv2.circle(frame, (cx, cy), 24, color, -1)
        cv2.drawMarker(frame, (cx, cy), (255, 255, 255), markerType=cv2.MARKER_CROSS, markerSize=20, thickness=2)
        cv2.rectangle(frame, (cx - 35, cy - 35), (cx + 35, cy + 35), (0, 255, 0), 1)
        cv2.putText(frame, f"TEST_PATTERN_{self.camera_id[:4].upper()}", (cx - 35, cy - 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1, cv2.LINE_AA)

        # HUD Overlay Notice
        now_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        cv2.putText(frame, f"CAM: {self.name} [{self.camera_id}]", (20, 35),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"SRC: {self.source_type.upper()} ({self.source_url[:40]})", (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1, cv2.LINE_AA)
        cv2.putText(frame, f"TIME: {now_str}", (20, 85),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        cv2.putText(frame, f"STATUS: SYNTHETIC FALLBACK (Camera not connected)", (20, 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 180, 255), 1, cv2.LINE_AA)
        if self.error_message:
            cv2.putText(frame, f"REASON: {self.error_message[:65]}", (20, 135),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 130, 255), 1, cv2.LINE_AA)

        return frame

    def get_latest_jpeg(self) -> Optional[bytes]:
        with self.lock:
            return self.latest_jpeg

    def get_inference_sample(self) -> Tuple[Optional[np.ndarray], bool]:
        """Samples frame if sample_interval has elapsed. Returns (frame, has_motion)."""
        now = time.time()
        if now - self.last_sample_time < self.sample_interval:
            return None, False

        with self.lock:
            if self.latest_frame is None:
                return None, False
            frame = self.latest_frame.copy()

        self.last_sample_time = now

        # Motion detection difference
        has_motion = True
        try:
            small_gray = cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY)
            if self.prev_gray_frame is not None:
                diff = cv2.absdiff(small_gray, self.prev_gray_frame)
                has_motion = float(np.mean(diff)) >= self.motion_threshold
            self.prev_gray_frame = small_gray
        except Exception:
            has_motion = True

        return frame, has_motion

    def get_status(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "name": self.name,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "preset_id": self.preset_id,
            "sample_fps": self.sample_fps,
            "actual_fps": round(self.actual_fps, 1),
            "status": self.status,
            "is_synthetic": self.is_synthetic,
            "error_message": self.error_message
        }


class MultiCameraManager:
    """Manages an arbitrary number of video ingestion workers (N-Camera Hub)."""

    def __init__(self):
        self.cameras: Dict[str, CameraDevice] = {}
        self.lock = threading.Lock()
        self.round_robin_index = 0

    def add_camera(
        self,
        camera_id: str,
        name: str,
        source_type: str,
        source_url: str,
        preset_id: str = "security",
        sample_fps: float = 1.0
    ) -> CameraDevice:
        """Registers and starts a new camera worker."""
        with self.lock:
            if camera_id in self.cameras:
                self.cameras[camera_id].stop()

            cam = CameraDevice(
                camera_id=camera_id,
                name=name,
                source_type=source_type,
                source_url=source_url,
                preset_id=preset_id,
                sample_fps=sample_fps
            )
            cam.start()
            self.cameras[camera_id] = cam
            logger.info(f"Registered camera '{name}' [ID: {camera_id}] (Total: {len(self.cameras)})")
            return cam

    def remove_camera(self, camera_id: str) -> bool:
        with self.lock:
            if camera_id in self.cameras:
                cam = self.cameras.pop(camera_id)
                cam.stop()
                logger.info(f"Removed camera '{cam.name}' [ID: {camera_id}]")
                return True
            return False

    def update_camera_preset(self, camera_id: str, preset_id: str) -> bool:
        with self.lock:
            if camera_id in self.cameras:
                self.cameras[camera_id].preset_id = preset_id
                return True
            return False

    def get_camera(self, camera_id: str) -> Optional[CameraDevice]:
        with self.lock:
            return self.cameras.get(camera_id)

    def get_all_cameras(self) -> List[CameraDevice]:
        with self.lock:
            return list(self.cameras.values())

    def get_all_status(self) -> List[Dict[str, Any]]:
        with self.lock:
            return [cam.get_status() for cam in self.cameras.values()]

    def get_next_sample(self) -> Optional[Tuple[str, np.ndarray, str, bool]]:
        with self.lock:
            cam_list = list(self.cameras.values())

        if not cam_list:
            return None

        total = len(cam_list)
        for i in range(total):
            idx = (self.round_robin_index + i) % total
            cam = cam_list[idx]
            frame, has_motion = cam.get_inference_sample()
            if frame is not None:
                self.round_robin_index = (idx + 1) % total
                return cam.camera_id, frame, cam.preset_id, has_motion

        return None

    def stop_all(self):
        with self.lock:
            for cam in self.cameras.values():
                cam.stop()
            self.cameras.clear()
        logger.info("All camera workers stopped.")


# Global camera manager instance
camera_manager = MultiCameraManager()
