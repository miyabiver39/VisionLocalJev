# Vision-Jev Guard Platform 🛡️📹

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Powered by: DiffusionGemma-Jev](https://img.shields.io/badge/Decision_Engine-DiffusionGemma--Jev_(DJev)-purple.svg)](https://github.com/Davipar/djev-dev)
[![Docker Image](https://img.shields.io/badge/Docker-compose-blue?logo=docker)](https://github.com/miyabiver39/VisionLocalJev/pkgs/container/visionlocaljev)
[![Tests](https://github.com/miyabiver39/VisionLocalJev/actions/workflows/tests.yml/badge.svg)](https://github.com/miyabiver39/VisionLocalJev/actions/workflows/tests.yml)
[![Python 3.12](https://img.shields.io/badge/Python-3.12-brightgreen.svg)](https://www.python.org/)

**Vision-Jev Guard Platform** は、監視カメラ（HLS / RTSP / YouTube / Webカメラ / HTTP JPEG）の映像フレームから、Google DeepMind の **DiffusionGemma** 上に構築された OSS の型付き判定システム **DJev（[`Davipar/djev-dev`](https://github.com/Davipar/djev-dev)）** の設問形式（Choice / Score / Noul）で状況判定を行うエッジ監視統合基盤です。

> [!NOTE]
> デフォルトの `DJEV_MODE=embedded` は、GPU なしで動かすための **CPU エミュレータ**（OpenCV 特徴量＋ルールベースのロジット計算）であり、DiffusionGemma モデル本体は実行しません。1 フレームあたりの処理時間は Ryzen 7 5700X・720p で約 10ms（`python scripts/benchmark_pipeline.py` で計測）です。

異常事態を検知した際は、**内包型SOP RAG（緊急初動手順ナレッジベース）**が状況に応じた対応マニュアルをリアルタイムに引き当て、**WebHook（Slack / Discord / 汎用JSON）**経由で外部警備システムへ自動通知します。

---

## 1. システムアーキテクチャ & パイプライン

```
  [ 監視カメラ群 (N台中継: HLS .m3u8 / RTSP / YouTube / Webカメラ) ]
                             │
                             ▼  動体差分スキップ (MOG2 背景差分・CPU 4ms)
  [ DiffusionGemma-Jev (DJev) Engine ]
  ・非自己回帰 離散テキスト拡散 (Discrete Text Diffusion, t=8 → 0)
  ・マルチモーダル画像テンソル ＋ 設問スキーマの直接融合
  ・Choice(確率分布), Score(切迫度), Noul(真偽値) を1パスで一括デノイズ確定
                             │
              ┌──────────────┴──────────────┐
              ▼                             ▼
  [ SOP RAG Recommendation ]      [ WebHook Dispatcher ]
  ・内包型緊急対応マニュアル検索       ・Slack / Discord
  ・外部RAGプロキシ連携 (Dify等)      ・汎用 JSON HTTP POST
              │                             │
              └──────────────┬──────────────┘
                             ▼
  [ FastAPI バックエンド & WebSocket 通信 ]
                             │
                             ▼
  [ 統合 WebUI (マルチカメラ同時監視 + DJev 拡散ステータス + パフォーマンスHUD) ]
```

---

## 2. 6つのドメインプリセット対応

現場の多様な監視ニーズに応えるため、YAML形式で自由に追加・カスタマイズ可能なプリセットを標準搭載しています：

| プリセットID | 名称 | 主な監視・判定対象（Choice） | 判定スコア / Noul |
|---|---|---|---|
| `security` | **防犯・立ち入り監視** | 正常通過 / 滞留徘徊 / 柵越え侵入 / 不審物放置 | 脅威度スコア / 警報発報要請 |
| `fire_disaster` | **火災・防災監視** | 正常 / 水蒸気（湯気）/ 黒煙 / 開放火炎 | 切迫度スコア / 避難判定 |
| `nursing_care` | **介護施設・見守り** | 正常就寝 / 起き上がり覚醒 / 夜間徘徊 / **転倒倒臥** | 介助緊急度 / スタッフ駆けつけ要請 |
| `river_flood` | **河川監視・水害警戒** | 平常水位 / 注意水位超過 / **越水堤防決壊** / 中州取り残され | 水害切迫度 / 避難指示・水防出動 |
| `factory_safety` | **工場・労働安全** | 安全作業 / 保護具未着用 / 危険域進入 / **作業員倒臥** | 労災リスク / ライン非常停止要請 |
| `railway_platform` | **駅ホーム・鉄道安全** | 安全待機 / 白線外側歩行 / 柵乗り越え / **線路転落** | 列車抑止切迫度 / 非常停止ボタン連動 |

---

## 3. Visual Example RAG（画像そのものの参照RAG）📸

テキストマニュアル（SOP）の引き当てだけでなく、**「画像そのものをベクトル登録し、現場の正常・異常の過去事例とミリ秒単位で照合する」** Visual Example RAG エンジンを標準搭載しています。

- **512次元マルチスケール特徴記述子**: Spatial 4x4 Grid HSV色相分布(192) + Sobel HOGエッジ勾配(160) + 空間テクスチャモーメント(160) をCPUで抽出（L2正規化、720p フレームで照合込み約 5ms）。
- **コサイン類似度 k-NN検索 & 異常度（Anomaly Score）自動算出**: 登録された「正常ベースライン画像」との乖離度と「過去の事故事例画像」との類似度から、リアルタイムにアノマリーを検知。
- **UIからの画像登録 & ギャラリー**: WebUIの「画像RAG」モーダルから、現場写真や異常事例画像をいつでもドラッグ＆ドロップで追加・管理可能。
- **6大プリセットの初期リファレンス画像（計12枚）を自動シード登録済み**。

---

## 4. クイックスタート

### 方法 A: Docker Compose での起動（推奨・ワンストップ）

```bash
docker compose up -d --build
```
起動後、ブラウザで **`http://localhost:8000`** にアクセスするとダッシュボードが表示されます。

> [!WARNING]
> デフォルトでは認証なしで `0.0.0.0:8000` に公開されます。LAN やインターネットから到達可能な環境では、必ず `AUTH_USERNAME` / `AUTH_PASSWORD` を設定して HTTP Basic 認証を有効化してください（例: `AUTH_USERNAME=admin AUTH_PASSWORD=change-me docker compose up -d`）。

### 方法 B: ローカルPython環境での起動

```bash
# 1. 仮想環境の作成
python -m venv venv
.\venv\Scripts\activate  # Linux/macOS: source venv/bin/activate

# 2. 依存関係のインストール
pip install -r requirements.txt

# 3. サーバー起動
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

---

## 5. 映像ソース（カメラ）の追加方法

WebUI上の **「+ カメラ追加 (RTSP/JPEG)」** から、様々なストリームを自由に登録できます：

1. **HLS ストリーミング（.m3u8）**:
   - 認証不要で100%確実に即座に再生可能。
   - サンプル: `https://test-streams.mux.dev/x36xhzz/x36xhzz.m3u8`
2. **YouTube 監視カメラ / ライブ配信**:
   - YouTube URL（`https://www.youtube.com/watch?v=...`）を入力。
   - ※ YouTube側のBotGuard遮断（`Sign in to confirm you're not a bot`）が発生する場合は、プロジェクトルートに `cookies.txt` を配置することで認証を通過可能（Docker の場合は `docker-compose.yml` の `cookies.txt` マウント行を有効化、または `YOUTUBE_COOKIE_FILE` でパスを指定）。`cookies.txt` は `.gitignore` 済みです。絶対にコミットしないでください。
3. **RTSP 監視カメラ**:
   - ネットワーク監視カメラのURL（`rtsp://user:pass@camera-ip:554/stream1`）を入力。
4. **ローカルWebカメラ**:
   - デバイス番号（`0`, `1` 等）を入力。

---

## 6. テスト & 単体動作検証スクリプト

```bash
pip install -r requirements-dev.txt
pytest
```

サーバーを起動せずに、コマンドラインから直接モデルやVisual RAGの動作を検証可能です：

```bash
# 1. DiffusionGemma-Jev (DJev) 画像フレーム＋設問JSONの直接判定テスト
python scripts/run_djev_example.py

# 2. Visual Example RAG (512次元ベクトル抽出 ＆ k-NN類似度照合) 単体テスト
python scripts/test_visual_rag.py
```

---

## 7. ドキュメント & リソース

- **[docs/sample_video_prompts.md](docs/sample_video_prompts.md)**:
  - Gemini Omni / Google Veo / Sora 等でテスト用動画（正常・異常の対比）を生成するためのプロンプト集。
- **[docs/qiita_article.md](docs/qiita_article.md)**:
  - Qiita投稿用の詳細技術解説記事（アーキテクチャ・否定文誤認の解消・パフォーマンス実測値）。

---

## 8. 環境変数一覧 (`.env`)

| 環境変数名 | デフォルト値 | 説明 |
|---|---|---|
| `CAMERA_SOURCE` | `synthetic` | 初期カメラソース（`synthetic` / `0` / HLS / RTSP / YouTube URL） |
| `DECISION_ENGINE` | `diffusion-gemma-jev` | 決定エンジン種別（`diffusion-gemma-jev`） |
| `DJEV_MODE` | `embedded` | `embedded`（CPU エミュレータ）または `remote`（[TypeSafe System One](https://docs.typesafe.ai/api) 互換サーバーへ `POST /v1/systemone`）。remote で失敗した場合はエラーとして扱い、エミュレータへは切り替えない |
| `DJEV_SERVER_URL` | `http://localhost:8765/v1/systemone` | `DJEV_MODE=remote` 時の接続先（`/v1/systemone` は省略可） |
| `DJEV_MODEL` | `jev-latest` | リクエストの `model`（例: `jev-latest`, `imajev-4b`, `qev:0.8b`） |
| `DJEV_API_KEY` | （空） | 設定時に `Authorization: Bearer <key>` を付与 |
| `DJEV_IMAGE_MODE` | `images` | 画像の渡し方。`images`（imajev: トップレベル `images` 配列）/ `state_content`（Qev: state を OpenAI 形式の content 配列に）/ `none`（TypeSafe Jev・Kev: テキストのみ） |
| `DJEV_TIMEOUT` | `10` | リモート推論のタイムアウト（秒） |
| `SAMPLE_FPS` | `1.0` | 推論サンプリングFPS（0.1〜10.0） |
| `ALERT_THRESHOLD` | `0.80` | アラートを発報するスコア/確信度の閾値 |
| `VISION_MODE` | `detector` | 映像状態抽出モード（`detector`: CPU CV / `mock`: デモ用シナリオ / `vlm`: Moondream2 等） |
| `RAG_SERVER_URL` | （空） | 外部 RAG サーバー URL。空の場合は内蔵 SOP RAG を使用 |
| `YOUTUBE_COOKIE_FILE` | （空） | YouTube 用 cookies.txt のパス（`cookies.txt` / `youtube_cookies.txt` も自動検出） |
| `AUTH_USERNAME` / `AUTH_PASSWORD` | （空） | 両方設定すると全ページ・API・映像・WebSocket に HTTP Basic 認証を適用（`/healthz` を除く） |
| `MAX_UPLOAD_IMAGE_BYTES` | `10485760` | Visual RAG へ登録する画像の最大サイズ（バイト） |

---

## 9. 判定エンジン API（TypeSafe System One 形式）

判定エンジンの入出力は [TypeSafe System One API](https://docs.typesafe.ai/api) の形式に揃えています。

- プリセット YAML の `questions` は TypeSafe の Question 形式（`type` / `instructions` / `criteria`）で、`label` は画面表示専用です。
- 判定結果は `answers`（`noul` / `choice` / `score` の Answer 形式）と `usage` で返ります（WebSocket の `decision.answers`、`/api/trigger_scenario` のレスポンス）。
- `DJEV_MODE=remote` では、TypeSafe 本家（`jev-latest`、テキストのみ）や互換モデル（imajev・Qev・Kev など）をそのまま使えます。
- リモート推論の失敗（接続失敗・タイムアウト・HTTP エラー・形式不正）は、画面に「判定エンジンエラー」として表示し、WebSocket に `decision_error` を送ります。`/api/trigger_scenario` は 502（タイムアウトは 504）を返します。CPU エミュレータへの自動切り替えは行いません。
- ローカル GPU で imajev を動かす手順は [docs/local_rocm_wsl.md](docs/local_rocm_wsl.md) を参照してください。

## 10. ライセンス

本プロジェクトは **MIT License** のもとで公開されています。商用・非商用問わず自由にご利用いただけます。
