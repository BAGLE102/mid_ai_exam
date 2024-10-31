# 使用 Google Colab 平台進行視網膜圖像分類模型訓練

本專案使用 Kaggle 數據集進行視網膜圖像分類，利用深度學習模型來自動分類視網膜影像，分類的對象涵蓋四種視網膜疾病類型。

---

## 數據集介紹
數據集來自 Kaggle，由專業醫院標註，包含以下四種類別：

- **CNV**：脈絡膜新生血管形成（具有新生血管膜及視網膜下液）
- **DME**：糖尿病性黃斑水腫（伴隨視網膜增厚的視網膜內液）
- **DRUSEN**：早期 AMD 中的玻璃疣
- **NORMAL**：正常視網膜（無視網膜液或水腫）

數據集文件大小約 5GB，包含超過 8 萬張圖像，並按類別分為訓練、測試、驗證三個文件夾。

---

## 開始使用

### 1. 掛載 Google Drive
首先，將 Google Drive 掛載到 Colab，以便存取和儲存模型及數據。

### 2. 下載數據集
在 Kaggle 中創建 API Token 並下載 `kaggle.json` 文件。然後，將 `kaggle.json` 移至 `.kaggle` 目錄，並使用 Kaggle API 下載數據集。

### 3. 數據處理
指定訓練及測試數據文件的路徑，以便後續加載數據。

定義數據處理函數，包含圖像加載、標籤解析、圖像縮放等操作。
