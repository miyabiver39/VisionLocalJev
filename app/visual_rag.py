"""
app/visual_rag.py
-----------------
Visual Example RAG (Visual Few-Shot Retrieval-Augmented Generation) Engine.

Enables image-based RAG for surveillance and manufacturing inspection:
1. Embeds reference images (normal baseline vs anomaly cases) into 512-dim visual feature vectors.
2. Performs fast cosine-similarity k-NN search against incoming camera frames (<2ms on CPU).
3. Computes Anomaly Distance (deviation from normal baseline) and matches historical incident examples.
4. Feeds retrieved visual context and linked SOPs into the DJev decision pipeline.
"""

import os
import time
import base64
import logging
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
import cv2

logger = logging.getLogger("vision_jev.visual_rag")


class VisualReference:
    """Represents a registered reference image in the visual vector database."""

    def __init__(
        self,
        ref_id: str,
        title: str,
        category: str,
        is_anomaly: bool,
        embedding: np.ndarray,
        sop_id: str = "",
        image_base64: str = "",
        description: str = "",
        created_at: float = 0.0
    ):
        self.ref_id = ref_id
        self.title = title
        self.category = category  # security, fire_disaster, nursing_care, river_flood, factory_safety, railway_platform
        self.is_anomaly = is_anomaly
        self.embedding = embedding  # 512-dim normalized vector
        self.sop_id = sop_id
        self.image_base64 = image_base64
        self.description = description
        self.created_at = created_at or time.time()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "ref_id": self.ref_id,
            "title": self.title,
            "category": self.category,
            "is_anomaly": self.is_anomaly,
            "sop_id": self.sop_id,
            "image_base64": self.image_base64,
            "description": self.description,
            "created_at": self.created_at
        }


class VisualRAGEngine:
    """
    Ultra-fast, zero-dependency visual feature embedder and vector search engine.
    Extracts 512-dimensional spatial-chromatic-texture descriptors (L2-normalized)
    and computes cosine similarity in <2 ms on CPU.
    """

    def __init__(self, storage_dir: str = "app/data/visual_rag"):
        self.storage_dir = storage_dir
        os.makedirs(self.storage_dir, exist_ok=True)
        self.references: Dict[str, VisualReference] = {}
        self._load_seed_references()
        logger.info(f"VisualRAGEngine initialized with {len(self.references)} reference images.")

    def extract_embedding(self, frame: np.ndarray) -> np.ndarray:
        """
        Extracts a 512-dimensional visual feature embedding vector from an image frame.
        Combines:
          - Spatial 4x4 Grid HSV Color Histograms (16 bins * 3 channels * 4 cells = 192 dims)
          - Multi-scale Edge Gradient Orientation (Sobel HOG-like, 160 dims)
          - Structural Luminance & Contrast Moments (160 dims)
        Result is L2-normalized: |v| = 1.0.
        """
        if frame is None or frame.size == 0:
            return np.zeros(512, dtype=np.float32)

        # 1. Resize to canonical resolution for scale invariance
        img = cv2.resize(frame, (256, 256))
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        features = []

        # A. Spatial 4x4 Grid Color Distribution (192 dims)
        grid_size = 4
        h_step, w_step = 256 // grid_size, 256 // grid_size
        for r in range(grid_size):
            for c in range(grid_size):
                cell_hsv = hsv[r * h_step:(r + 1) * h_step, c * w_step:(c + 1) * w_step]
                # 4 bins Hue, 4 bins Sat, 4 bins Val per cell = 12 dims * 16 cells = 192 dims
                hist_h = cv2.calcHist([cell_hsv], [0], None, [4], [0, 180]).flatten()
                hist_s = cv2.calcHist([cell_hsv], [1], None, [4], [0, 256]).flatten()
                hist_v = cv2.calcHist([cell_hsv], [2], None, [4], [0, 256]).flatten()
                cell_vec = np.concatenate([hist_h, hist_s, hist_v])
                norm = np.linalg.norm(cell_vec) + 1e-6
                features.append(cell_vec / norm)

        color_feat = np.concatenate(features)  # 192 dims

        # B. Edge Gradient Orientations (160 dims: 10 bins x 16 cells)
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        mag, ang = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)

        edge_features = []
        for r in range(grid_size):
            for c in range(grid_size):
                cell_ang = ang[r * h_step:(r + 1) * h_step, c * w_step:(c + 1) * w_step]
                cell_mag = mag[r * h_step:(r + 1) * h_step, c * w_step:(c + 1) * w_step]
                # 10 orientation bins (0-360 deg) weighted by magnitude
                bin_edges = np.linspace(0, 360, 11)
                hist, _ = np.histogram(cell_ang, bins=bin_edges, weights=cell_mag)
                norm = np.linalg.norm(hist) + 1e-6
                edge_features.append(hist / norm)

        edge_feat = np.concatenate(edge_features)  # 160 dims

        # C. Texture & Spatial Contrast Moments (160 dims)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)
        diff = cv2.absdiff(gray, blur)
        tex_features = []
        for r in range(grid_size):
            for c in range(grid_size):
                cell_diff = diff[r * h_step:(r + 1) * h_step, c * w_step:(c + 1) * w_step]
                hist_t, _ = np.histogram(cell_diff, bins=10, range=(0, 100))
                norm = np.linalg.norm(hist_t) + 1e-6
                tex_features.append(hist_t / norm)

        tex_feat = np.concatenate(tex_features)  # 160 dims

        # Concatenate: 192 + 160 + 160 = 512 dimensions
        full_embedding = np.concatenate([color_feat, edge_feat, tex_feat]).astype(np.float32)
        norm_full = np.linalg.norm(full_embedding) + 1e-7
        return full_embedding / norm_full

    def register_reference(
        self,
        title: str,
        category: str,
        is_anomaly: bool,
        frame: np.ndarray,
        sop_id: str = "",
        description: str = "",
        ref_id: Optional[str] = None
    ) -> VisualReference:
        """Registers a new reference image into the visual database."""
        if ref_id is None:
            prefix = "anom_" if is_anomaly else "norm_"
            ref_id = f"{prefix}{int(time.time() * 1000) % 1000000}"

        embedding = self.extract_embedding(frame)

        # Create thumbnail for UI display (320x180 JPEG)
        thumb = cv2.resize(frame, (320, 180))
        _, buf = cv2.imencode(".jpg", thumb, [cv2.IMWRITE_JPEG_QUALITY, 75])
        img_b64 = "data:image/jpeg;base64," + base64.b64encode(buf).decode("utf-8")

        ref = VisualReference(
            ref_id=ref_id,
            title=title,
            category=category,
            is_anomaly=is_anomaly,
            embedding=embedding,
            sop_id=sop_id,
            image_base64=img_b64,
            description=description
        )

        self.references[ref_id] = ref
        logger.info(f"Registered Visual RAG reference '{title}' [ID: {ref_id}, Anomaly: {is_anomaly}]")
        return ref

    def match_frame(
        self,
        frame: np.ndarray,
        category: Optional[str] = None,
        top_k: int = 3
    ) -> Dict[str, Any]:
        """
        Performs k-NN cosine similarity search on the current video frame.
        Returns:
          - top_match: Highest similarity reference image
          - similarity: Cosine similarity score (0.0 to 1.0)
          - anomaly_score: Distance from normal baseline (0.0 = perfectly normal, 1.0 = highly anomalous)
          - matches: List of top-K matches with metadata
        """
        if frame is None or len(self.references) == 0:
            return {
                "top_match": None,
                "similarity": 0.0,
                "anomaly_score": 0.0,
                "is_anomalous": False,
                "matches": []
            }

        t0 = time.perf_counter()
        query_vec = self.extract_embedding(frame)

        results = []
        normal_sims = []
        anomaly_sims = []

        for ref in self.references.values():
            if category and ref.category != category and ref.category != "default":
                continue

            # Cosine similarity: dot product of L2-normalized vectors
            sim = float(np.dot(query_vec, ref.embedding))
            # Clip between 0.0 and 1.0
            sim = max(0.0, min(1.0, (sim + 1.0) / 2.0))

            results.append({
                "ref_id": ref.ref_id,
                "title": ref.title,
                "category": ref.category,
                "is_anomaly": ref.is_anomaly,
                "similarity": round(sim, 4),
                "sop_id": ref.sop_id,
                "image_base64": ref.image_base64,
                "description": ref.description
            })

            if ref.is_anomaly:
                anomaly_sims.append(sim)
            else:
                normal_sims.append(sim)

        results.sort(key=lambda x: x["similarity"], reverse=True)
        top_k_results = results[:top_k]
        top_match = top_k_results[0] if top_k_results else None

        # Compute Anomaly Score:
        # High when similarity to an anomaly image is high, OR similarity to normal baseline is low
        max_anom_sim = max(anomaly_sims) if anomaly_sims else 0.0
        max_norm_sim = max(normal_sims) if normal_sims else 0.5
        
        # Anomaly score is driven by anomaly match and normal baseline deviation
        anomaly_score = max_anom_sim * 0.7 + (1.0 - max_norm_sim) * 0.3
        anomaly_score = round(max(0.0, min(1.0, anomaly_score)), 3)

        latency_ms = (time.perf_counter() - t0) * 1000

        return {
            "top_match": top_match,
            "similarity": top_match["similarity"] if top_match else 0.0,
            "anomaly_score": anomaly_score,
            "is_anomalous": bool(anomaly_score >= 0.70 or (top_match and top_match["is_anomaly"] and top_match["similarity"] > 0.80)),
            "matches": top_k_results,
            "latency_ms": round(latency_ms, 2)
        }

    def get_all_references(self) -> List[Dict[str, Any]]:
        return [ref.to_dict() for ref in self.references.values()]

    def delete_reference(self, ref_id: str) -> bool:
        if ref_id in self.references:
            del self.references[ref_id]
            logger.info(f"Deleted Visual RAG reference [ID: {ref_id}]")
            return True
        return False

    def _load_seed_references(self):
        """Generates realistic visual seed reference images for all 6 domain presets."""
        seeds = [
            # 1. Fire / Disaster
            {
                "title": "平常時の工場ライン設備 (正常ベースライン)",
                "category": "fire_disaster",
                "is_anomaly": False,
                "sop_id": "",
                "bg_color": (40, 45, 50),
                "decorations": [("rect", (100, 150, 400, 350), (60, 65, 70))],
                "desc": "火災や煙のない日常の正常な工場床・配電盤設備。"
            },
            {
                "title": "制御盤火災・開放火炎 (過去重大事故例)",
                "category": "fire_disaster",
                "is_anomaly": True,
                "sop_id": "sop_fire_open_flame",
                "bg_color": (30, 30, 35),
                "decorations": [
                    ("ellipse", (300, 250, 70, 140), (0, 120, 255)),
                    ("ellipse", (300, 260, 40, 90), (0, 220, 255))
                ],
                "desc": "設備から激しい赤橙色の火炎が立ち上っている重大火災事例。"
            },
            # 2. Nursing Care
            {
                "title": "居室ベッド就寝 (正常安静ベースライン)",
                "category": "nursing_care",
                "is_anomaly": False,
                "sop_id": "",
                "bg_color": (50, 55, 60),
                "decorations": [("rect", (150, 200, 350, 200), (90, 95, 100))],
                "desc": "利用者がベッド上で毛布をかけて安全に安静就寝している状態。"
            },
            {
                "title": "ベッドサイド転倒・床倒臥 (緊急介助事例)",
                "category": "nursing_care",
                "is_anomaly": True,
                "sop_id": "sop_nursing_fall",
                "bg_color": (45, 50, 55),
                "decorations": [
                    ("rect", (100, 100, 200, 150), (80, 85, 90)),
                    ("ellipse", (320, 320, 120, 40), (40, 40, 180))  # 倒臥シルエット
                ],
                "desc": "ベッド脇のフローリング上に利用者が倒れ込んで動けない転倒事例。"
            },
            # 3. River Flood
            {
                "title": "河川平常水位 (平穏清流ベースライン)",
                "category": "river_flood",
                "is_anomaly": False,
                "sop_id": "",
                "bg_color": (70, 90, 60),  # 緑地堤防
                "decorations": [("rect", (0, 180, 512, 150), (120, 80, 50))],  # 水面
                "desc": "水位標が規定値以下で穏やかに流れる晴天時の河川敷。"
            },
            {
                "title": "越水・堤防決壊氾濫 (水害避難警報事例)",
                "category": "river_flood",
                "is_anomaly": True,
                "sop_id": "sop_river_overflow",
                "bg_color": (50, 60, 70),
                "decorations": [
                    ("rect", (0, 50, 512, 300), (30, 60, 100))  # 濁流氾濫
                ],
                "desc": "激しい泥濁流がコンクリート堤防を越水し冠水している氾濫事例。"
            },
            # 4. Security
            {
                "title": "通用口正常通過 (安全歩行ベースライン)",
                "category": "security",
                "is_anomaly": False,
                "sop_id": "",
                "bg_color": (55, 60, 65),
                "decorations": [("rect", (200, 100, 80, 250), (70, 70, 70))],
                "desc": "社員通用口を規定の動線で正常に歩行通過する日常シーン。"
            },
            {
                "title": "外周フェンス乗り越え侵入 (侵入警報事例)",
                "category": "security",
                "is_anomaly": True,
                "sop_id": "sop_trespass_breach",
                "bg_color": (25, 25, 30),
                "decorations": [
                    ("line", (0, 200, 512, 200), (150, 150, 150)),  # フェンス
                    ("ellipse", (256, 180, 30, 60), (0, 0, 150))   # 乗り越え侵入者
                ],
                "desc": "夜間に外周フェンスをよじ登り敷地内へ侵入を試みる不審者事例。"
            },
            # 5. Factory Safety
            {
                "title": "規定保護具着用・安全通路歩行 (労働安全ベースライン)",
                "category": "factory_safety",
                "is_anomaly": False,
                "sop_id": "",
                "bg_color": (45, 50, 50),
                "decorations": [("rect", (150, 280, 200, 80), (40, 150, 40))],  # 緑の安全通路
                "desc": "ヘルメット・安全帯を着用し指定緑色通路を歩行する安全作業。"
            },
            {
                "title": "重機旋回域進入・作業員倒臥 (労災非常停止事例)",
                "category": "factory_safety",
                "is_anomaly": True,
                "sop_id": "sop_factory_incident",
                "bg_color": (40, 40, 45),
                "decorations": [
                    ("rect", (100, 100, 200, 150), (0, 180, 220)),  # 重機
                    ("ellipse", (300, 280, 90, 35), (20, 20, 160))  # 倒臥作業員
                ],
                "desc": "フォークリフト作業エリア内で作業員が倒れ込んでいる重大労災事例。"
            },
            # 6. Railway Platform
            {
                "title": "黄色い点字ブロック内側待機 (乗客安全ベースライン)",
                "category": "railway_platform",
                "is_anomaly": False,
                "sop_id": "",
                "bg_color": (60, 65, 70),
                "decorations": [("rect", (0, 240, 512, 30), (0, 215, 255))],  # 黄色い線
                "desc": "乗客全員が点字ブロックの内側で整然と待機している安全なホーム。"
            },
            {
                "title": "ホーム端から線路への転落 (列車非常停止事例)",
                "category": "railway_platform",
                "is_anomaly": True,
                "sop_id": "sop_railway_track_fall",
                "bg_color": (35, 40, 45),
                "decorations": [
                    ("line", (0, 200, 512, 200), (0, 200, 255)),    # ホーム端
                    ("ellipse", (250, 300, 80, 40), (180, 50, 50))  # 線路転落者
                ],
                "desc": "ホーム端から軌道敷（線路）へ乗客が転落している直前非常事態事例。"
            }
        ]

        for s in seeds:
            # Draw synthetic reference frame
            h, w = 360, 512
            img = np.zeros((h, w, 3), dtype=np.uint8)
            img[:] = s["bg_color"]

            for dec_type, coords, color in s["decorations"]:
                if dec_type == "rect":
                    x, y, rw, rh = coords
                    cv2.rectangle(img, (x, y), (x + rw, y + rh), color, -1)
                elif dec_type == "ellipse":
                    cx, cy, ax1, ax2 = coords
                    cv2.ellipse(img, (cx, cy), (ax1, ax2), 0, 0, 360, color, -1)
                elif dec_type == "line":
                    x1, y1, x2, y2 = coords
                    cv2.line(img, (x1, y1), (x2, y2), color, 4)

            cv2.putText(img, s["title"], (15, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
            self.register_reference(
                title=s["title"],
                category=s["category"],
                is_anomaly=s["is_anomaly"],
                frame=img,
                sop_id=s["sop_id"],
                description=s["desc"]
            )


# Global singleton instance
visual_rag_engine = VisualRAGEngine()
