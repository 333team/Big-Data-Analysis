# Big-Data-Analysis
# 🎓 教育大數據：學習歷程分析儀表板

這是一個基於 Python 與 Streamlit 開發的互動式資料分析平台，專注於探討 **「記憶鞏固 (Memory Consolidation)」** 與 **「適度困難 (Desirable Difficulties)」** 理論在數位學習中的實證表現。

本系統鎖定學習後的 **0 ~ 24 小時** 關鍵區間，透過高解析度的時間切分與 AI 機器學習模型，分析學生的答題正確率、反應時間與學習行為模式，並提供即時的學習風險診斷。

## ✨ 核心功能

*   **📊 互動式儀表板**：即時計算有效樣本數、活躍學生數、平均正確率與答題耗時。
*   **📉 記憶鞏固曲線 (Consolidation Curve)**：
    *   使用線性時間軸 (Linear Scale) 呈現 0-24 小時的變化。
    *   支援依照「題目難易度」分層，觀察不同難度下的記憶保留狀況。
*   **⏱️ 認知負荷分析**：透過答題時間 (Response Time) 的變化，推測學生在不同延遲時間下的提取難度。
*   **👥 學習族群差異**：
    *   比較「高分組」與「潛力組」的行為模式。
    *   分析不同「知識向度」（如：擷取訊息 vs. 發展解釋）的遺忘趨勢。
*   **🤖 AI 風險預測**：
    *   內建 Random Forest (隨機森林) 模型，預測學生是否需要補救教學。
    *   提供特徵重要性 (Feature Importance) 與混淆矩陣分析。
    *   **單一學生診斷工具**：輸入參數即可獲得即時風險評估。

## 📂 專案結構

為了確保程式在本地端與 Streamlit Cloud 上都能正常運作，請維持以下目錄結構：

```text
src/
├── app.py                      # 主程式碼
├── requirements.txt            # Python 套件清單
├── NotoSansTC-Regular.ttf      # 中文字型檔 (必須與 app.py 同層)
└── resource/                   # 資料放置目錄
    └── anonymized_file0115.csv # 預設讀取的資料檔
```

## 🚀 快速開始 (本地端執行)

### 1. 安裝環境
確保您已安裝 Python 3.8+。建議建立虛擬環境：

```bash
# 建立虛擬環境
python -m venv venv

# 啟動虛擬環境 (Windows)
venv\Scripts\activate

# 啟動虛擬環境 (Mac/Linux)
source venv/bin/activate
```

### 2. 安裝依賴套件
建立 `requirements.txt` (內容見下方)，並執行安裝：

```bash
pip install -r requirements.txt
```

**requirements.txt 內容：**
```text
streamlit==1.50.0
pandas==2.3.3
statsmodels==0.14.6
scikit-learn==1.8.0
seaborn==0.13.2
matplotlib==3.10.7
```

### 3. 準備字型與資料
1.  **下載字型**：請至 Google Fonts 下載 [Noto Sans TC](https://fonts.google.com/specimen/Noto+Sans+TC)，將 `NotoSansTC-Regular.ttf` 放入專案根目錄。
2.  **準備資料**：將您的 CSV 檔案重新命名為 `anonymized_file0115.csv` 並放入 `resource/` 資料夾中。

### 4. 啟動應用程式
```bash
streamlit run app.py
```

## 📊 資料格式說明

本系統預期 CSV 檔案需包含以下欄位（欄位名稱需完全一致）：

| 欄位名稱 | 說明 |
| :--- | :--- |
| `學生姓名去識別化` | 學生唯一 ID |
| `任務派發時間` | 格式：YYYY-MM-DD HH:MM:SS |
| `學生首次送出答案的時間點` | 格式：YYYY-MM-DD HH:MM:SS |
| `首次答題正確率` | 數值 (0~1 或 0~100) |
| `首次答題時間（秒）` | 數值 (秒數) |
| `難易度` | 類別 (如：易, 中, 難) |
| `任務名稱` | (選填) 任務標題 |
| `擷取訊息正確率` | (選填) 知識向度欄位 |
| `發展解釋正確率` | (選填) 知識向度欄位 |

## ☁️ 部署至 Streamlit Cloud

若您要將此專案部署到網路上，請注意以下幾點：

1.  **上傳 GitHub**：將上述所有檔案（包含 `.ttf` 字型檔與 `resource` 資料夾）推送到 GitHub Repository。
2.  **字型問題**：程式碼已內建 `get_chinese_font()` 函式，會自動讀取根目錄下的 `NotoSansTC-Regular.ttf`，解決雲端環境中文亂碼問題。
3.  **路徑設定**：程式碼使用相對路徑偵測 (`os.path.join`)，確保能正確讀取 `resource/` 下的檔案。

## 💡 分析方法論

*   **時間分箱 (Binning)**：
    *   **0~6小時**：每 1 小時切分，捕捉黃金學習窗口。
    *   **6~24小時**：每 3 小時切分，觀察晝夜節律與疲勞效應。
*   **資料清洗**：
    *   **IQR 過濾**：自動移除答題時間過短（<1秒）或過長（> Q3 + 1.5*IQR）的異常值。
    *   **範圍鎖定**：僅保留練習延遲在 24 小時內的樣本。

## 📝 License

此專案僅供教育數據分析競賽與學術研究使用。