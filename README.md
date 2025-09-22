# Raspberry Pi 智慧廣告看板範例

此專案示範如何在樹莓派上整合 USB 攝影機、dlib 人臉辨識、SQLite 與 Flask，達成匿名化會員識別與廣告推播。流程如下：

1. 攝影機擷取影像並透過 dlib 取得 128 維臉部特徵。
2. 將特徵向量雜湊成匿名 ID，寫入 SQLite 並記錄首次辨識時間。
3. 讀取會員歷史消費紀錄（預設 5 筆模擬資料），依模板生成廣告訊息。
4. Flask Web 頁面每兩秒輪詢最新廣告，於平板或螢幕上即時顯示。

## 專案架構

```
├── README.md
├── requirements.txt
├── src/
│   └── pi_kiosk/
│       ├── advertising.py      # 廣告模板與訊息生成
│       ├── database.py         # SQLite 初始化與資料存取
│       ├── detection.py        # dlib 臉部辨識與匿名 ID 生成
│       ├── flask_app.py        # Flask 服務與命令列介面
│       └── pipeline.py         # 整體流程協調（辨識、資料庫、廣告）
└── tests/
    ├── test_advertising.py
    ├── test_database.py
    └── test_pipeline.py
```

## 安裝指南

### 1. 系統需求

- 樹莓派 4B/5（建議 4GB RAM 以上）
- USB 攝影機（支援 UVC）
- Raspberry Pi OS (Bullseye) / Debian 12
- Python 3.10 以上

### 2. 必要套件

```bash
sudo apt update
sudo apt install -y build-essential cmake python3-dev libopenblas-dev liblapack-dev libx11-dev libgtk-3-dev
```

安裝 Python 套件：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

`requirements.txt` 內容涵蓋 `dlib`, `opencv-python`, `Flask` 等核心套件。樹莓派上若安裝 `dlib` 遇到困難，可先安裝 `libatlas-base-dev` 等依賴或改用預編譯輪檔。

### 3. 下載 dlib 模型

請下載以下檔案並置於 `models/` 目錄：

- [shape_predictor_68_face_landmarks.dat](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2)
- [dlib_face_recognition_resnet_model_v1.dat](http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2)

解壓縮後結構如下：

```
models/
├── dlib_face_recognition_resnet_model_v1.dat
└── shape_predictor_68_face_landmarks.dat
```

若需要加速或較佳準確率，可另外下載 `mmod_human_face_detector.dat` 並於設定中啟用 CNN 偵測模式。

### 4. 離線訓練流程

1. 準備資料夾結構 `data/faces/<member_id>/image.jpg`，詳見 `data/faces/README.md`。每人建議 5~10 張不同角度的臉部照片。
2. 於開發機執行：

   ```bash
   PYTHONPATH=src python scripts/build_encodings.py --input data/faces --output models/known_faces.npz
   PYTHONPATH=src python scripts/train_classifier.py --input models/known_faces.npz --output models/face_classifier.pkl
   ```

   產生的 `models/face_classifier.pkl` 會同時儲存平均特徵向量與辨識閾值。若需匿名化，可在第二步加上 `--hash-labels`。
3. 將 `models/face_classifier.pkl` 與 dlib 預訓練檔一併複製到 Raspberry Pi 的 `models/` 目錄，即可直接推論。

## 執行方式

### 1. 模擬模式（無攝影機，展示流程）

```bash
source .venv/bin/activate
python -m pi_kiosk.flask_app --db-path data/demo.db --simulate-members member-001 member-002
```

瀏覽 `http://<Raspberry-Pi-IP>:8000/`，頁面會輪播不同會員的模板廣告，可用 `/api/simulate` POST 端點觸發特定 ID：

```bash
curl -X POST http://<IP>:8000/api/simulate -H 'Content-Type: application/json' -d '{"member_id": "member-003"}'
```

### 2. 攝影機即時辨識

```bash
source .venv/bin/activate
python -m pi_kiosk.flask_app --camera --db-path data/kiosk.db --model-dir models --classifier models/face_classifier.pkl
```

若未提供 `--classifier`，系統會自動尋找 `models/face_classifier.pkl`，找不到時回到匿名雜湊模式。

參數說明：

- `--camera-index`：攝影機編號（預設 0）。
- `--frame-width`, `--frame-height`：可選擇設定影像解析度。
- `--cooldown-seconds`：於 `PipelineConfig` 中可微調同一 ID 觸發間隔（預設 2 秒）。
- `--idle-reset-seconds`：閒置多久後回復「等待辨識中…」，設定為 0 可停用。

程式會啟動背景執行緒讀取攝影機並進行辨識，Flask 頁面即時更新廣告。

## SQLite 結構與示範資料

- `members(id TEXT PRIMARY KEY, first_seen TEXT)`
- `transactions(id INTEGER, member_id TEXT, item TEXT, amount REAL, timestamp TEXT)`

首次啟動時會自動寫入 5 筆模擬消費紀錄：

| Member | 商品 | 金額 | 時間 |
| ------ | ---- | ---- | ---- |
| member-001 | 有機牛奶 | 85 | 2024-05-02 09:30 |
| member-001 | 手工優格 | 120 | 2024-05-15 14:15 |
| member-002 | 燕麥片 | 65 | 2024-04-28 11:42 |
| member-003 | 冷萃咖啡 | 90 | 2024-05-01 08:20 |
| member-003 | 義式麵包 | 55 | 2024-05-12 18:45 |

廣告模板會根據最近一次購買時間與金額決定折扣策略（80 元以上 9 折，否則 85 折）。

## 測試

使用標準函式庫 `unittest` 撰寫測試，並模擬辨識流程：

```bash
PYTHONPATH=src python -m unittest discover -s tests
```

## 進一步擴充

- **硬體優化**：可改用 `libcamera` 或 CSI 攝影機，提高畫質與效能。
- **資料庫整合**：將 SQLite 改為遠端 API 或 MQTT 以對接賣場後端。
- **廣告排程**：在 `advertising.generate_message` 中加入更多商品推薦策略或 GPT 生成文案。
- **隱私保護**：現行僅儲存特徵雜湊，若需進一步匿名化可加入 Differential Privacy 或清除策略。
