# 使用 Google Colab 平台進行視網膜圖像分類模型訓練

本專案使用 Kaggle 數據集進行視網膜圖像分類，利用深度學習模型來自動分類視網膜影像，分類的對象涵蓋四種視網膜疾病類型。

## 數據集介紹
數據集來自 Kaggle，由專業醫院標註，包含以下四種類別：
- **CNV**：脈絡膜新生血管形成（具有新生血管膜及視網膜下液）
- **DME**：糖尿病性黃斑水腫（伴隨視網膜增厚的視網膜內液）
- **DRUSEN**：早期 AMD 中的玻璃疣
- **NORMAL**：正常視網膜（無視網膜液或水腫）

數據集文件大小約 5GB，包含超過 8 萬張圖像，並按類別分為訓練、測試、驗證三個文件夾。

## 開始使用

### 1. 掛載 Google Drive
```python
from google.colab import drive
drive.mount('/content/gdrive/')
