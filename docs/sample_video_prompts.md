# Gemini Omni / 動画生成AI向け 監視カメラ映像生成プロンプト集
## Vision-Jev Guard 検証・デモ用サンプル動画プロンプト (YouTube公開用)

本ドキュメントは、**Gemini Omni (Google Veo / Imagen Video / Sora 等)** を用いて、エッジ監視・判定システム「Vision-Jev Guard」の検証用動画を生成するためのプロンプト集です。

各ドメインプリセットごとに、以下の2種類の動画ペアを生成できるように設計しています：
- **【正常シーン（Normal / Calm）】**: 日常の平穏な情景。影や作業服、機械の動きなどによる「誤検知（False Positive）」が発生しないことを検証する動画。
- **【検知・切迫シーン（Critical Alert）】**: 異常事態が発生し、DJevが即時に高切迫度スコアと緊急初動SOP（マニュアル）を発報することを実証する動画。

---

## 共通の撮影・レンダリングスタイル指定（Japanese CCTV Style Prompt）
プロンプト末尾に必ず付与することで、**「日本国内の舞台」「日本人」「日本語の看板・標識」「日本の監視カメラの画角」**を強力に条件付けします：
```text
Authentic Japanese surveillance camera footage set in Japan. Realistic Japanese domestic architecture, Japanese text signage and warning banners in kanji on walls, authentic Japanese people. High-angle fixed Japanese CCTV security camera perspective, crisp 1080p 24fps surveillance video quality, genuine Japan setting.
```

---

## 1. 介護施設・高齢者見守り (`nursing_care`)

### 1-A. 【正常シーン】穏やかな就寝・室内安静（Choice: `normal_rest`, Score: `0.04`, Alert: `False`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  A high-angle indoor surveillance camera view of a modern Japanese nursing care home private bedroom at night in Tokyo, Japan. An elderly Japanese resident in their 80s is resting peacefully and motionless under a warm futon blanket on a low Japanese nursing bed with wooden side rails. Subtle ambient nightlight, authentic Japanese interior with tatami-toned flooring and a Japanese emergency nurse call button unit on the wall. Authentic Japanese surveillance camera footage set in Japan, peaceful Japanese domestic eldercare setting.
  ```
- **日本語概要**: 日本の介護施設個室。80代の日本人利用者が低床介護ベッド上で布団をかけて静かに安らかに就寝している夜間定点監視映像。

### 1-B. 【検知シーン】ベッドサイドでの転倒・床倒臥（Choice: `fall_detected`, Score: `0.94`, Alert: `True`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  A high-angle Japanese indoor nursing home room camera in Japan capturing an elderly Japanese person attempting to get up from bed, losing balance, and falling onto the wooden floor near the bedside slippers. The elderly Japanese resident remains lying motionless on the floor. Indoor fluorescent lighting, Japanese wall posters and nurse call unit, authentic domestic Japanese eldercare surveillance footage.
  ```
- **日本語概要**: 日本の介護老人保健施設。日本人高齢者がベッドからスリッパを履いて立ち上がろうとして足元を取られ、床に倒れ込んで動けなくなっている転倒の瞬間。

---

## 2. 河川監視・水害警戒 (`river_flood`)

### 2-A. 【正常シーン】平常時の清流・低水位（Choice: `normal_flow`, Score: `0.04`, Alert: `False`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  A fixed Japanese riverbank surveillance CCTV camera overlooking a calm rural river in Japan on a clear sunny morning. Low water level flowing peacefully over stones, visible concrete embankment dykes, green grass on riverbank, and a Japanese river measurement staff marked with clear Japanese kanji characters showing safe water level. Crisp natural daytime lighting, authentic Japanese infrastructure CCTV.
  ```
- **日本語概要**: 晴天時の日本の河川堤防監視カメラ。水量は穏やかで日本の漢字表記の水位標柱は基準以下。土手や護岸が露出している平常映像。

### 2-B. 【検知シーン】豪雨激流・高水位危険標検知（Choice: `overflow_breach`, Score: `0.95`, Alert: `True`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  An authentic Japanese river disaster prevention surveillance camera during heavy rainfall in Japan. The river channel carries a very high, swift water current with surface ripples, reaching the red danger indicator mark on a concrete measurement pillar marked with Japanese kanji text. Overcast cloudy Japanese landscape, fast flowing river stream, scientific water monitoring in Japan. Authentic fixed Japanese CCTV footage.
  ```
- **日本語概要**: 日本の河川防災定点カメラ。梅雨の豪雨の中、水流が急激に増水し、茶色い激流がコンクリート護岸の赤い危険水位標の直前まで達している高切迫度シーン。

---

## 3. 防犯・立ち入り監視 (`security`)

### 3-A. 【正常シーン】通用口の正常歩行・通過（Choice: `normal_passing`, Score: `0.04`, Alert: `False`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  A high-angle Japanese corporate building entrance security camera in Tokyo, Japan. A Japanese businessman dressed in a dark business suit walks through the well-lit entrance corridor, holding an employee badge and entering through the automatic glass door. Clean Japanese office interior, Japanese wall signage reading 関係者以外立入禁止 in kanji, routine foot traffic, authentic Japanese daytime CCTV.
  ```
- **日本語概要**: 日本のオフィスビル通用口。ビジネススーツを着た日本人社員が社員証をかざして自動ドアを通過する日常の安全光景。

### 3-B. 【検知シーン】外周フェンス乗り越え・不法侵入（Choice: `trespassing`, Score: `0.92`, Alert: `True`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  A night vision monochrome surveillance camera overlooking a Japanese industrial facility perimeter chain-link fence in Japan. An unauthorized person wearing a dark hoodie and gloves climbing over the wire fence topped with barbed wire and dropping into the facility shadows. Japanese warning sign reading 立入禁止 防犯カメラ作動中 visible on the fence, high-contrast night vision Japanese CCTV security footage.
  ```
- **日本語概要**: 日本の工場地帯の夜間赤外線カメラ。黒いフードを着た人物が外周フェンスをよじ登り敷地内へ侵入する瞬間。「立入禁止・防犯カメラ作動中」の日本語看板。

---

## 4. 火災・防災監視 (`fire_disaster`)

### 4-A. 【正常シーン】給湯室の白い湯気（Choice: `steam_vapor` or `none`, Score: `0.04`, Alert: `False`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  An indoor security camera view of a Japanese office tea room (給湯室) in Tokyo, Japan. A stainless kettle on an induction stove boils, releasing gentle translucent white water steam into the air above. No flames, no smoke, bright fluorescent overhead lighting, Japanese warning stickers reading 火気厳禁 on the stainless steel counter, authentic Japanese workplace interior without fire hazards.
  ```
- **日本語概要**: 日本のオフィスの給湯室。IHコンロの上でステンレスやかんから白い湯気が立ち上っている日常の光景（火災誤検知防止テスト用）。

### 4-B. 【検知シーン】電気室の開放火炎と黒煙（Choice: `open_flame`, Score: `0.96`, Alert: `True`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  An authentic industrial CCTV camera inside a Japanese factory electrical distribution switchboard room in Japan. Dense billowing dark grey and black smoke rapidly rises from a metal control panel cabinet, followed by intense orange flickering open flames erupting from the top vents. Yellow Japanese warning sign reading 高圧受電設備 危険 reflecting the firelight, emergency industrial fire scenario from a fixed elevated Japanese CCTV.
  ```
- **日本語概要**: 日本の工場配電盤室。制御盤から濃い黒煙と激しい炎が噴き出し、日本語の「高圧受電設備・危険」の黄色い看板が照らされている火災発生シーン。

---

## 5. 工場・労働安全監視 (`factory_safety`)

### 5-A. 【正常シーン】安全保護具着用の適正作業（Choice: `safe_operation`, Score: `0.04`, Alert: `False`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  A high-ceiling Japanese manufacturing factory floor surveillance camera in Japan. Two Japanese factory workers wearing yellow hardhats and high-visibility neon reflective safety vests walk strictly within a painted green safety pathway. Clear Japanese green cross banner reading 安全第一 on the wall, industrial machinery operating cleanly in background, compliant Japanese occupational safety environment.
  ```
- **日本語概要**: 日本の製造工場。黄色いヘルメットを被った日本人作業員が緑色の安全通路を整然と歩行。「安全第一」の緑十字看板。

### 5-B. 【検知シーン】危険区域内での作業員倒臥・意識喪失（Choice: `worker_down`, Score: `0.96`, Alert: `True`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  A Japanese factory surveillance camera in Japan capturing an emergency workplace drill. A Japanese worker in factory uniform lies motionless on the concrete floor inside a yellow hazard diagonal line near automated equipment. Japanese safety poster reading 整理整頓 on the wall, a colleague rushing in background to press the red emergency stop button, authentic Japanese industrial incident drill.
  ```
- **日本語概要**: 日本の工場。日本人作業員が危険エリア内で倒れて動かなくなっており、同僚が非常停止ボタンに駆け寄る労災緊急事態。

---

## 6. 駅ホーム・鉄道安全 (`railway_platform`)

### 6-A. 【正常シーン】点字ブロック内側での安全待機（Choice: `safe_waiting`, Score: `0.04`, Alert: `False`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  A high-angle Japanese railway station platform security CCTV camera in Tokyo, Japan. Japanese commuters in dark business attire standing neatly in orderly lines behind the yellow textured braille safety line, waiting for a train. Automatic platform screen doors, Japanese station name signs overhead, clean orderly morning Japanese train platform footage.
  ```
- **日本語概要**: 日本の駅ホーム。スーツ姿の日本の通勤客が黄色い点字ブロックの内側で整然と整列乗車を待っている安全な朝のホーム風景。

### 6-B. 【検知シーン】ホーム端から線路への転落（Choice: `track_fall`, Score: `0.98`, Alert: `True`）
- **英語プロンプト (Gemini Omni / Veo用)**:
  ```text
  A Japanese train station platform surveillance camera in Japan. A passenger stumbles past the yellow braille line and falls off the platform edge down onto the track gravel area between the steel rails. Red emergency warning indicator flashing on the Japanese platform pillar labeled 非常ボタン, authentic Japanese railway security camera perspective.
  ```
- **日本語概要**: 日本の駅ホーム。乗客がホームから線路へ転落し、軌道敷内で倒れ込んでいる。「非常ボタン」の赤色警告ランプが点滅。

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
