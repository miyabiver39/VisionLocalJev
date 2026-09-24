# Gemini Omni / 動画生成AI向け 監視カメラ映像生成プロンプト集
## Vision-Jev Guard 検証・デモ用サンプル動画プロンプト (YouTube公開用)

本ドキュメントは、**Gemini Omni (Google Veo / Imagen Video / Sora 等)** を用いて、エッジ監視・判定システム「Vision-Jev Guard」の検証用動画を生成するためのプロンプト集です。

各ドメインプリセットごとに、以下の2種類の動画ペアを生成できるように設計しています：
- **【正常シーン（Normal / Calm）】**: 日常の平穏な情景。影や作業服、機械の動きなどによる「誤検知（False Positive）」が発生しないことを検証する動画。
- **【検知・切迫シーン（Critical Alert）】**: 異常事態が発生し、DJevが即時に高切迫度スコアと緊急初動SOP（マニュアル）を発報することを実証する動画。

---

## 共通の撮影・レンダリングスタイル指定（Style Prompt）
プロンプト末尾に付与することで、リアルな監視カメラ（CCTV）の質感を高めます：
```text
Style parameters: High-angle static surveillance camera (CCTV footage), wide-angle lens with subtle barrel distortion, authentic fixed security perspective, realistic lighting and shadows, CCTV digital timestamp and camera label overlay in the upper corner, 1080p 24fps surveillance video quality, no cinematic handheld motion.
```

---

## 1. 介護施設・高齢者見守り (`nursing_care`)

### 1-A. 【正常シーン】穏やかな就寝・室内安静（Choice: `normal_rest`, Score: `0.04`, Alert: `False`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  A realistic high-angle security camera view of a clean, softly lit nursing home private bedroom at night. An elderly person is sleeping peacefully and motionless under blankets on a low nursing bed with side rails raised. Soft amber nightlight in the corner of the room. No sudden movements, serene and safe atmosphere. Static CCTV camera footage with green timestamp in top-left corner.
  ```
- **日本語概要**: 介護居室の夜間定点監視。利用者がベッド柵の中で毛布をかけて穏やかに就寝しており、室内は静かで安全。

### 1-B. 【検知シーン】ベッドサイドでの転倒・床倒臥（Choice: `fall_detected`, Score: `0.94`, Alert: `True`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  A high-angle nursing home indoor CCTV camera capturing an elderly resident attempting to stand up from the bed, losing balance, tripping, and falling heavily onto the wooden floor near the bedside. The resident remains lying motionless on the floor beside the slippers. Realistic slow movement of the fall, natural indoor fluorescent lighting, security camera angle looking down from the ceiling corner, authentic surveillance footage.
  ```
- **日本語概要**: 高齢者がベッドから立ち上がろうとしてバランスを崩し、床に倒れ込んで動けなくなっている転倒の瞬間。

---

## 2. 河川監視・水害警戒 (`river_flood`)

### 2-A. 【正常シーン】平常時の清流・低水位（Choice: `normal_flow`, Score: `0.04`, Alert: `False`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  A fixed riverbank surveillance camera overlooking a calm rural river on a clear sunny day. The water level is low and flowing smoothly over riverbed stones. Visible dry concrete embankments, green grass along the river trail, and an empty water gauge pole clearly showing normal baseline water levels. High-angle static CCTV framing with crisp daytime lighting.
  ```
- **日本語概要**: 晴天時の堤防監視カメラ。水量は穏やかで水位標は基準以下。川原や堤防斜面が完全に露出している平常映像。

### 2-B. 【検知シーン】越水・堤防決壊と濁流氾濫（Choice: `overflow_breach`, Score: `0.95`, Alert: `True`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  A stormy river surveillance camera during heavy torrential rainfall. The river has swollen dramatically, turned into a rapid muddy brown torrent, and is overflowing the concrete dyke embankments. Muddy floodwaters inundate the surrounding paved road and park benches. Debris and tree branches rushing past in turbulent currents. Dramatic emergency flood situation from a fixed elevated CCTV perspective.
  ```
- **日本語概要**: 豪雨時の河川激流。茶色の濁流がコンクリート堤防を越水し、道路や遊歩道へと冠水・氾濫している緊急水害映像。

---

## 3. 防犯・立ち入り監視 (`security`)

### 3-A. 【正常シーン】通用口の正常歩行・通過（Choice: `normal_passing`, Score: `0.04`, Alert: `False`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  A high-angle commercial building entrance security camera. An employee dressed in smart business attire walks casually through the well-lit entrance corridor, holding an employee badge and entering through the automatic glass door. Clean, routine foot traffic, daylight coming through windows, calm and secure corporate environment.
  ```
- **日本語概要**: ビル入口のセキュリティカメラ。社員が身分証を持って普通に自動ドアを通過して入館する日常光景。

### 3-B. 【検知シーン】外周フェンス乗り越え・不法侵入（Choice: `trespassing`, Score: `0.92`, Alert: `True`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  Night vision infrared security camera capturing an unauthorized person wearing a dark hoodie and gloves climbing aggressively over an 8-foot chain-link perimeter fence topped with barbed wire. The person leaps down into the restricted facility courtyard and darts into building shadows. High contrast night vision monochrome CCTV footage, slight motion blur, urgent security intrusion scenario.
  ```
- **日本語概要**: 夜間赤外線カメラ。暗い服の人物が工場の外周フェンスをよじ登り、敷地内へ飛び降りて侵入する瞬間。

---

## 4. 火災・防災監視 (`fire_disaster`)

### 4-A. 【正常シーン】給湯室の湯気・水蒸気（Choice: `steam_vapor` or `none`, Score: `0.04`, Alert: `False`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  A security camera view of an industrial office tea room/kitchen. A kettle on an induction stove boils, releasing gentle, translucent white water steam into the air above. No flames, no black smoke, bright fluorescent overhead lighting, stainless steel counters. Safe and normal kitchen scene without fire hazards.
  ```
- **日本語概要**: 給湯室でやかんから白い湯気が立ち上っている様子。黒煙や炎はなく、誤検知防止のテストに最適。

### 4-B. 【検知シーン】電気室の開放火炎と黒煙（Choice: `open_flame`, Score: `0.96`, Alert: `True`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  Indoor CCTV camera inside a factory electrical distribution room. Dense, billowing black smoke rapidly rises from a metal control panel cabinet, followed by intense orange flickering open flames erupting from the top vents of the machine. Smoke accumulates across the ceiling creating a hazing layer. Emergency industrial fire breakout, static high-angle camera framing.
  ```
- **日本語概要**: 工場電気室の制御盤から濃い黒煙が噴き出し、激しい炎が立ち上る火災発生の瞬間。

---

## 5. 工場・労働安全監視 (`factory_safety`)

### 5-A. 【正常シーン】安全保護具着用の適正作業（Choice: `safe_operation`, Score: `0.04`, Alert: `False`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  High-ceiling factory manufacturing floor surveillance camera. Two industrial workers wearing yellow hardhats, high-visibility neon reflective safety vests, and protective boots are walking within painted green pedestrian safety walkways. Machining centers operating behind clear safety barriers in background. Compliant occupational safety environment.
  ```
- **日本語概要**: 工場内の定点カメラ。ヘルメットと反射ベストを着用した作業員が安全通路を歩行し、規定通り作業している様子。

### 5-B. 【検知シーン】危険区域内での作業員倒臥・意識喪失（Choice: `worker_down`, Score: `0.96`, Alert: `True`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  Surveillance camera overlooking a warehouse heavy machinery forklift loading zone marked with yellow hazard stripes. A worker in overalls suddenly stumbles and collapses unconscious onto the concrete floor inside the active forklift path, remaining completely still. Yellow rotating emergency warning beacon reflecting nearby. Urgent industrial accident scene.
  ```
- **日本語概要**: 重機搬送エリアの床に作業員が突然倒れ込み、意識を失って動かなくなっている労災緊急事態。

---

## 6. 駅ホーム・鉄道安全 (`railway_platform`)

### 6-A. 【正常シーン】点字ブロック内側での安全待機（Choice: `safe_waiting`, Score: `0.04`, Alert: `False`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  Railway platform ceiling-mounted surveillance camera looking down at passenger boarding area. Commuters standing neatly behind the tactile yellow braille line waiting for a train. Automatic half-height platform screen doors in front of the tracks. Calm morning commute, natural platform lighting, completely orderly and safe waiting conditions.
  ```
- **日本語概要**: 駅ホームカメラ。乗客が黄色い点字ブロックの内側で整然と列車を待っている安全な朝のホーム風景。

### 6-B. 【検知シーン】ホーム端から線路への転落（Choice: `track_fall`, Score: `0.98`, Alert: `True`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  High-angle security camera on a train station platform. A passenger stumbles past the yellow safety line and accidentally falls off the platform edge down onto the ballast and railway tracks below. The person struggles to get up between the steel rails as overhead station lights glare. Extreme railway hazard scenario captured on static CCTV.
  ```
- **日本語概要**: 乗客がバランスを崩してホームから線路へ転落し、軌道敷内で倒れ込んでいる直前非常事態。

---

---

## 7. アップロード済み YouTube 検証動画一覧 (UIワンクリック対応)

ユーザー様により YouTube へアップロードされた検証用動画一覧です。WebUIの「カメラ追加」モーダルからワンクリックで自動入力・即時監視可能です：

| # | YouTube URL | 対象ドメインプリセット | 概要 |
|---|---|---|---|
| **#1** | [`https://www.youtube.com/watch?v=IS98_Tzwl3c`](https://www.youtube.com/watch?v=IS98_Tzwl3c) | `security` (防犯・立ち入り) | 不審者徘徊・侵入検知 |
| **#2** | [`https://www.youtube.com/watch?v=NKHab4poTok`](https://www.youtube.com/watch?v=NKHab4poTok) | `fire_disaster` (火災・防災) | 電気室黒煙・火炎検知 |
| **#3** | [`https://www.youtube.com/watch?v=lYqbEQ93RBw`](https://www.youtube.com/watch?v=lYqbEQ93RBw) | `nursing_care` (介護見守り) | ベッドサイド高齢者転倒 |
| **#4** | [`https://www.youtube.com/watch?v=CRwce8_7t_o`](https://www.youtube.com/watch?v=CRwce8_7t_o) | `river_flood` (河川水害) | 増水・高水位危険標検知 |
| **#5** | [`https://www.youtube.com/watch?v=DW0F_5YUEVs`](https://www.youtube.com/watch?v=DW0F_5YUEVs) | `factory_safety` (工場労働安全) | 危険域進入・作業員倒臥 |
| **#6** | [`https://www.youtube.com/watch?v=Q3Aj3ynUyk0`](https://www.youtube.com/watch?v=Q3Aj3ynUyk0) | `railway_platform` (駅ホーム鉄道安全) | ホーム端から線路への転落 |
| **#7** | [`https://www.youtube.com/watch?v=hC214WzegXw`](https://www.youtube.com/watch?v=hC214WzegXw) | `security` (防犯・外周フェンス) | 夜間フェンス乗り越え |

> [!IMPORTANT]
> **YouTubeの公開設定について**:
> YouTube側の設定が「**非公開 (Private)**」のままですと、外部クライアント（yt-dlp / OpenCV）から映像ストリームを取得できません。
> YouTube Studioにて公開設定を **「限定公開 (Unlisted)」** または **「公開 (Public)」** に変更していただくことで、WebUIから確実にストリーミング再生・リアルタイム推論が可能となります。

---

## 8. 「越水・堤防決壊」等の動画生成が失敗する原因と回避策 (RAI Filter)

### 失敗の原因: Google Responsible AI (RAI) セーフティフィルター
Google DeepMind の動画生成モデル（Veo / ImageFX / VideoFX）には、厳格な倫理・安全フィルターが搭載されています。
以下の単語や表現は「**自然災害による人的危機・恐怖感の煽動（Disaster & Catastrophic Harm）**」として自動検出され、生成が遮断（`raiMediaFilteredCount: 1`）されます：
- ❌ 遮断されやすい表現: `floodwaters`, `overflowing dykes`, `dam breach`, `torrential calamity`, `muddy brown torrent`, `debris rushing past`

### 解決策: 学術シミュレーション・環境モニタリングへのリフレーム
危機感を煽る災害描写ではなく、**「気象観測所の定点カメラ映像」「水理工学モデルシミュレーション」「河川高水位標の超過」**といった客観的・学術的な表現に言い換えることで、安全フィルターを通過させることができます：

- ⭕ **安全な推奨プロンプト（河川増水・高水位危険シーン）**:
  ```text
  An environmental river monitoring station camera during autumn rainfall. The river channel carries a very high, swift water current with surface ripples, reaching the red high-water danger line on the concrete pillar. Overcast cloudy sky, fast flowing river stream, scientific environmental water monitoring. High-angle static surveillance camera perspective, realistic CCTV fixed framing.
  ```

---

## 9. Veo 3.1 Fast (Gemini API) 経由での動画自動生成スクリプト

Gemini API（Google Generative Language API）の `models/veo-3.1-fast-generate-preview` を用いれば、WebUIを使わずにPythonコードから直接高品質な監視カメラ動画（1080p MP4）を一括生成可能です。

### バッチ生成スクリプトの実行方法

```bash
# 環境変数 GEMINI_API_KEY が設定されている状態で実行
python scripts/generate_all_veo_videos.py
```

- 全6ドメイン（計12シーン）のプロンプトが自動で順次Veo APIへ投入されます。
- レンダリングが完了したMP4動画は、ローカルの `app/static/sample_videos/` に自動保存されます。
- WebUIの「カメラ追加」から「動画ファイル」として即座にローカル読み込み可能です。
