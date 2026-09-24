import time
import re
import os
import json
import logging
import requests
import numpy as np
from typing import Dict, Any, List, Optional
from pydantic import BaseModel

logger = logging.getLogger("vision_jev.rag")

# Short / function words must never count as keyword evidence (e.g. "a" is a substring of almost everything)
MIN_TOKEN_LEN = 3
STOPWORDS = {
    "the", "and", "are", "for", "with", "near", "from", "into", "onto", "over",
    "this", "that", "has", "have", "been", "being", "was", "were", "any", "area",
}
NEGATIONS = ["no ", "not ", "non-", "never ", "without ", "clear of ", "free of ", "no visible "]


def _contains_unnegated(text: str, phrase: str) -> bool:
    """True if `phrase` occurs in `text` at least once without a preceding negation word."""
    pos = 0
    while True:
        idx = text.find(phrase, pos)
        if idx == -1:
            return False
        prefix = text[max(0, idx - 25):idx]
        if not any(neg in prefix for neg in NEGATIONS):
            return True
        pos = idx + len(phrase)


def _keyword_token_hit(kw_lower: str, q_tokens: set, query_lower: str) -> bool:
    """Word-level match: every word of the keyword matches a query token exactly or as a stem prefix
    (e.g. keyword "loiter" matches "loitering"). Negated occurrences are ignored."""
    kw_words = [w for w in re.findall(r"\w+", kw_lower) if len(w) >= MIN_TOKEN_LEN]
    if not kw_words:
        return False
    for w in kw_words:
        hits = [t for t in q_tokens if t == w or t.startswith(w)]
        if not any(_contains_unnegated(query_lower, t) for t in hits):
            return False
    return True


class SOPDocument(BaseModel):
    id: str
    category: str
    title: str
    trigger_keywords: List[str]
    priority: str  # CRITICAL | HIGH | MEDIUM | LOW
    summary: str
    procedure_steps: List[str]
    emergency_contacts: Dict[str, str]


# Default Preloaded SOP Knowledge Base
INITIAL_SOP_KNOWLEDGE = [
    SOPDocument(
        id="sop_trespass_breach",
        category="security",
        title="外周柵越え・不法侵入発生時の一時対応手順",
        trigger_keywords=["climb", "fence", "trespass", "barrier", "breach", "outer", "intrusion", "侵入", "柵越え"],
        priority="CRITICAL",
        summary="外周フェンス越えおよび無許可立ち入り検知時の緊急対処マニュアル",
        procedure_steps=[
            "1. 該当カメラのライブ映像を全画面で固定監視し、侵入者の現在地・服装・持ち物を記録する。",
            "2. 現地警備員および常駐巡回スタッフへインカムで即座に緊急コード（Code Red: 外周侵入）を発令する。",
            "3. 侵入経路（南側フェンス/資材搬入口）のゲートを電子施錠し、施設内への逃走経路を遮断する。",
            "4. 状況に応じて110番通報を行い、警察官の現場到着を要請する。"
        ],
        emergency_contacts={
            "中央警備室": "内線 1100 / 03-XXXX-1100",
            "所轄警察署": "110 (警備担当: 03-XXXX-0110)",
            "施設防災センター": "内線 1119"
        }
    ),
    SOPDocument(
        id="sop_loitering_suspicious",
        category="security",
        title="不審者滞留・徘徊検知時の声掛け・確認手順",
        trigger_keywords=["loiter", "linger", "suspicious", "hoodie", "dark clothes", "徘徊", "不審者", "滞留"],
        priority="HIGH",
        summary="敷地境界付近での長時間の滞留・物色行動検知時の対応マニュアル",
        procedure_steps=[
            "1. 滞留者の滞在時間および周囲の観察行動（写真撮影・鍵穴物色等）の有無をログに記録する。",
            "2. 巡回スタッフを該当ポイントへ派遣し、「何かお困りですか」と自然な声掛け確認を行う。",
            "3. 退去要請に応じない場合、警備室より外部拡声スピーカーにて警告アナウンスを放送する。",
            "4. 防犯カメラのプリセットを追跡モードに切り替え、ナンバープレート等の特徴を記録する。"
        ],
        emergency_contacts={
            "中央警備室": "内線 1100",
            "巡回警備班リーダー": "内線 1105"
        }
    ),
    SOPDocument(
        id="sop_unattended_object",
        category="security",
        title="不審物・放置バッグ検知時の危険回避手順",
        trigger_keywords=["backpack", "bag", "unattended", "object", "left", "box", "package", "不審物", "放置"],
        priority="HIGH",
        summary="非常口・通路付近に放置された所有者不明の荷物検知時の対応マニュアル",
        procedure_steps=[
            "1. 放置物は絶対に触れず、周囲半径5m以内をカラーコーンまたは立入禁止テープで保全する。",
            "2. 巻き戻し録画（直近30分）を確認し、荷物を置いた人物の特定・置き忘れか否かを判定する。",
            "3. 配線や異臭、時計音等の危険兆候がある場合は直ちに館内非常放送を実施しフロア避難を誘導する。",
            "4. 警察署爆発物処理班または専門対応部署へ通報する。"
        ],
        emergency_contacts={
            "警備司令室": "内線 1100",
            "危険物・施設保全課": "内線 2240"
        }
    ),
    SOPDocument(
        id="sop_fire_open_flame",
        category="fire_disaster",
        title="火災・開放火炎検知時の緊急避難・初期消火手順",
        trigger_keywords=["fire", "flame", "open flame", "flicker", "burning", "火災", "炎", "火炎"],
        priority="CRITICAL",
        summary="明確な火炎発生時の初期消火および全員避難誘導マニュアル",
        procedure_steps=[
            "1. 直ちに火災報知器（発信機）を作動させ、全館避難アナウンスを起動する。",
            "2. 炎が天井に達していない場合のみ、備え付けの粉末消火器による初期消火を試みる。",
            "3. 119番通報を実施（火災場所、燃えている物、負傷者の有無を伝達）。",
            "4. 防火扉・防火シャッターを遠隔閉鎖し、排煙設備を手動起動する。",
            "5. 全員の屋外指定避難場所（中央広場）への避難完了を確認する。"
        ],
        emergency_contacts={
            "消防署 (119番)": "119 / 03-XXXX-0119",
            "自衛消防隊本部": "内線 1199",
            "夜間施設管理者": "090-XXXX-XXXX"
        }
    ),
    SOPDocument(
        id="sop_dark_smoke",
        category="fire_disaster",
        title="黒煙・煙火災検知時の給気停止・探索手順",
        trigger_keywords=["smoke", "dark smoke", "black smoke", "billow", "煙", "黒煙"],
        priority="CRITICAL",
        summary="電気室・倉庫等での発煙検知時の排煙・延焼防止マニュアル",
        procedure_steps=[
            "1. 空気調和・給気ファンを即座に停止し、煙の他フロアへの拡散を防止する。",
            "2. 防煙マスクを着用した防災要員2名1組で出火元・発煙設備の現場確認を実施。",
            "3. 電気室の場合、配電盤の主ブレーカーを安全に遮断し二次感電・ショートを予防する。",
            "4. 消防署へ通報し、煙が濃密な場合は迷わず避難誘導を開始する。"
        ],
        emergency_contacts={
            "消防署 (119番)": "119",
            "電気保安技術員": "内線 3320"
        }
    ),
    SOPDocument(
        id="sop_steam_vapor",
        category="fire_disaster",
        title="水蒸気・湯気誤報確認・配管点検手順",
        trigger_keywords=["steam", "vapor", "kettle", "white vapor", "湯気", "水蒸気", "配管漏れ"],
        priority="LOW",
        summary="白煙・蒸気発生時の誤報防止および給湯設備点検手順",
        procedure_steps=[
            "1. 現場の給湯設備、空調加湿器、ボイラー配管の点検を目視確認する。",
            "2. 蒸気漏れがある場合はバルブを閉止し、設備保守担当へ修理を依頼する。",
            "3. 火災報知器の誤作動を防止するため、現場の換気を実施する。"
        ],
        emergency_contacts={
            "設備保全課": "内線 4410",
            "中央管理室": "内線 1000"
        }
    ),
    SOPDocument(
        id="sop_nursing_fall",
        category="nursing_care",
        title="高齢者・入所者転倒検知時の救急救護手順",
        trigger_keywords=["fall", "fallen", "lying on floor", "collapse", "tripped", "転倒", "倒臥"],
        priority="CRITICAL",
        summary="居室または廊下での転倒・床倒臥検知時の迅速救護および外傷確認マニュアル",
        procedure_steps=[
            "1. ナースステーション常駐スタッフへPHS一斉呼出（コード・フォール）を発信。",
            "2. 最寄りの介護職員が30秒以内に居室へ急行し、意識・呼吸・頭部外傷の有無を確認する。",
            "3. 骨折の疑いがある場合は無理に動かさず、バイタルサイン（血圧・SpO2）を測定。",
            "4. 意識混濁・頭部打撲の場合は直ちに協力医療機関または119番へ連絡・救急搬送を要請。"
        ],
        emergency_contacts={
            "ナースステーション直通": "PHS 201 / 内線 2001",
            "協力提携クリニック": "03-XXXX-8820",
            "救急隊 (119番)": "119"
        }
    ),
    SOPDocument(
        id="sop_nursing_wandering",
        category="nursing_care",
        title="夜間徘徊・居室離床検知時の見守り・誘導手順",
        trigger_keywords=["wandering", "unsteady", "night walk", "pacing", "sitting up", "徘徊", "離床", "覚醒"],
        priority="HIGH",
        summary="深夜帯の単独歩行・転倒リスク者離床検知時の声掛け・安否確認マニュアル",
        procedure_steps=[
            "1. 該当フロアの夜勤巡回スタッフへインカム通知（○○号室様 離床確認）。",
            "2. 非常口および階段扉の自動電子ロックが作動していることを遠隔モニターで確認。",
            "3. スタッフが笑顔で穏やかに「こんばんは、お水をお持ちしましょうか」と声掛けし居室へ誘導。",
            "4. 水分補給・トイレ誘導を行い、ベッドサイドのマットセンサーを再セットする。"
        ],
        emergency_contacts={
            "夜間巡回当番": "PHS 205",
            "夜勤主任": "PHS 200"
        }
    ),
    SOPDocument(
        id="sop_river_overflow",
        category="river_flood",
        title="河川越水・堤防決壊警戒時の避難指示・水防出動手順",
        trigger_keywords=["overflow", "breach", "inundation", "dyke break", "submerged", "stranded", "越水", "氾濫", "決壊", "中州"],
        priority="CRITICAL",
        summary="水位センサー及びカメラ画像解析による越水兆候・氾濫発生時の緊急水防マニュアル",
        procedure_steps=[
            "1. 水害警戒レベル4（避難指示）またはレベル5（緊急安全確保）を自治体防災無線で一斉発令。",
            "2. 水防団へ緊急招集命令を発出し、陸閘門閉止および大型土のう設置現場へ配備。",
            "3. 中州・河川敷に取り残された民間人を確認した場合、消防水難救助隊およびヘリ救助を即時要請。",
            "4. 浸水想定区域内の住民へ高所避難（垂直避難）の徹底をエリアメールで配信。"
        ],
        emergency_contacts={
            "河川国道事務所・防災統括": "042-XXX-9900",
            "市役所災害対策本部": "内線 8800",
            "水難救助隊 (119番)": "119"
        }
    ),
    SOPDocument(
        id="sop_factory_incident",
        category="factory_safety",
        title="工場作業員倒臥・重機危険進入時のライン非常停止手順",
        trigger_keywords=["worker down", "unconscious", "danger zone", "forklift", "collapsed worker", "倒臥", "危険区域", "重機"],
        priority="CRITICAL",
        summary="重機作業エリア侵入・作業員意識喪失検知時の自動ラインカット・現場救護マニュアル",
        procedure_steps=[
            "1. 該当製造ラインおよびAGV（無人搬送車）の自動インターロック非常停止を作動させる。",
            "2. 工場内非常灯（赤色回転灯）を点灯させ、安全衛生管理者および医務室へ急行要請。",
            "3. 二次災害防止のため、設備主電源および高圧エア配管を元バルブにて遮断。",
            "4. AEDおよび救急担架を携行した救護ペアを直ちに現場到着させ初期蘇生を実施。"
        ],
        emergency_contacts={
            "工場医務室": "内線 5555",
            "中央防災・安全環境課": "内線 5510",
            "救急要請 (119番)": "119"
        }
    ),
    SOPDocument(
        id="sop_railway_track_fall",
        category="railway_platform",
        title="駅ホーム線路転落・列車非常停止ボタン連動手順",
        trigger_keywords=["track fall", "fallen onto tracks", "on the rails", "track intrusion", "転落", "線路立入", "非常停止"],
        priority="CRITICAL",
        summary="線路内転落者・ホーム柵突破検知時の列車防護無線・緊急抑止手順",
        procedure_steps=[
            "1. 信号システムへ直結し、該当駅進入中の全列車へATS非常停止信号（停止現示）を自動送出。",
            "2. ホーム上およびコンコースの非常停止ボタン表示器を全点灯させ、接近案内を抑止放送に切り替え。",
            "3. 駅係員2名がホーム下退避スペースへの誘導ロープを投入し、安全確認を肉眼で実施。",
            "4. 運転指令所へ線路閉鎖を宣言し、電力区へ第三軌条/架線の送電停止を要請。"
        ],
        emergency_contacts={
            "総合運転指令所": "直通 03-XXXX-7700",
            "駅事務室当直主任": "内線 7001",
            "電力指令室 (送電停止)": "内線 7720"
        }
    )
]


class RAGEngine:
    """Hybrid RAG engine supporting built-in SOP knowledge retrieval and external RAG proxy."""

    def __init__(self, external_rag_url: Optional[str] = None):
        self.external_rag_url = external_rag_url or os.getenv("RAG_SERVER_URL", "").strip()
        self.documents: Dict[str, SOPDocument] = {doc.id: doc for doc in INITIAL_SOP_KNOWLEDGE}
        logger.info(f"Initialized Embedded SOP RAG with {len(self.documents)} preloaded documents.")
        if self.external_rag_url:
            logger.info(f"External RAG proxy enabled: {self.external_rag_url}")

    def search_sop(self, query_text: str, category: Optional[str] = None) -> Dict[str, Any]:
        """
        Retrieves the most relevant emergency procedure for the detected situation.
        Uses external RAG server if configured, else uses embedded semantic-lexical hybrid ranking.
        """
        start_time = time.time()
        
        # 1. Try external RAG if URL is configured
        if self.external_rag_url:
            try:
                ext_result = self._query_external_rag(query_text, category)
                if ext_result:
                    latency = (time.time() - start_time) * 1000.0
                    ext_result["latency_ms"] = round(latency, 2)
                    ext_result["source"] = "external_rag"
                    return ext_result
            except Exception as e:
                logger.warning(f"External RAG query failed ({e}). Falling back to embedded SOP knowledge.")

        # 2. Embedded Hybrid Search
        doc, score = self._embedded_hybrid_search(query_text, category)
        latency = (time.time() - start_time) * 1000.0

        if doc is None:
            return {
                "source": "embedded_rag",
                "matched": False,
                "latency_ms": round(latency, 2),
                "sop": None
            }

        return {
            "source": "embedded_rag",
            "matched": True,
            "relevance_score": round(float(score), 3),
            "latency_ms": round(latency, 2),
            "sop": doc.model_dump()
        }

    def _embedded_hybrid_search(self, query_text: str, category: Optional[str]) -> tuple[Optional[SOPDocument], float]:
        """Performs lexical and token-overlap relevance scoring against SOP documents."""
        query_lower = query_text.lower()
        q_tokens = {
            t for t in re.findall(r"\w+", query_lower)
            if len(t) >= MIN_TOKEN_LEN and t not in STOPWORDS
        }
        best_doc = None
        highest_score = 0.0

        for doc in self.documents.values():
            if category and doc.category != category:
                continue

            score = 0.0
            # Keyword weight: full phrase hit (non-negated) > word-level stem hit
            for kw in doc.trigger_keywords:
                kw_lower = kw.lower()
                if _contains_unnegated(query_lower, kw_lower):
                    score += 3.0
                elif _keyword_token_hit(kw_lower, q_tokens, query_lower):
                    score += 1.0

            # Priority boost
            if doc.priority == "CRITICAL":
                score *= 1.25
            elif doc.priority == "HIGH":
                score *= 1.1

            if score > highest_score:
                highest_score = score
                best_doc = doc

        if highest_score < 1.0:
            return None, 0.0

        normalized_score = min(1.0, highest_score / 10.0)
        return best_doc, normalized_score

    def _query_external_rag(self, query_text: str, category: Optional[str]) -> Optional[Dict[str, Any]]:
        """Sends payload to external RAG HTTP endpoint."""
        resp = requests.post(
            self.external_rag_url,
            json={"query": query_text, "category": category},
            timeout=3.0
        )
        if resp.status_code == 200:
            data = resp.json()
            return {"matched": True, "sop": data}
        return None

    def add_document(self, doc: SOPDocument):
        self.documents[doc.id] = doc
        logger.info(f"Added new SOP document: {doc.title} ({doc.id})")

    def list_documents(self) -> List[Dict[str, Any]]:
        return [doc.model_dump() for doc in self.documents.values()]

    def get_document(self, doc_id: str) -> Optional[Dict[str, Any]]:
        doc = self.documents.get(doc_id)
        return doc.model_dump() if doc else None


# Global RAG engine instance
rag_engine = RAGEngine()
