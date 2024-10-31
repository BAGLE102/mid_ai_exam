# It seems there was an error in the previous cell where the string literal was not properly closed.
# I will correct the code in that cell and proceed with creating the notebook.

# Fixing the previous code snippet and completing the notebook content
notebook_content = {
    "cells": [
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "# Google Colab 平台訓練模型案例\n",
                "南華大學 跨領域-人工智慧期中報告\n",
                "\n",
                "## 數據集介紹\n",
                "\n",
                "![image](https://github.com/11024244/mid/blob/main/jpg/01.png)\n",
                "\n",
                "數據集來自 **Kaggle**，品質很高，由知名醫院的專業人員嚴格審核標註，如圖所示數據有 4 種類別：\n",
                "\n",
                "- **CNV**：具有新生血管膜和相關視網膜下液的脈絡膜新血管形成\n",
                "- **DME**：糖尿病性黃斑水腫與視網膜增厚相關的視網膜內液\n",
                "- **DRUSEN**：早期 AMD 中存在多個玻璃疣\n",
                "- **NORMAL**：視網膜正常，沒有任何視網膜液或水腫\n",
                "\n",
                "![image](https://github.com/11024244/mid/blob/main/jpg/02.png)\n",
                "\n",
                "檔案大小約為 5GB，包含 8 萬多張圖像，分為訓練、測試、驗證三個資料夾，每個資料夾按照種類不同分成 4 個子資料夾，其次是具體圖像檔。\n",
                "\n",
                "## 數據集下載\n",
                "\n",
                "**掛載資料夾**：\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {
                "colab_type": "code",
                "id": "c151e1ca",
                "outputId": "d23c38b9-13b3-404c-9df5-39397c62b7e5",
                "collapsed": False
            },
            "outputs": [],
            "source": [
                "from google.colab import drive\n",
                "\n",
                "# 掛載 Google Drive，以便讀取和存儲數據\n",
                "drive.mount('/content/gdrive/')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "按照提示進行驗證，結果如下：\n",
                "\n",
                "![image](https://github.com/11024244/mid/blob/main/jpg/03.png)\n",
                "\n",
                "**Kaggle 資料下載**：\n",
                "\n",
                "創建 **kaggle** 帳戶並下載 **kaggle.json** 檔。創建帳戶這裡就不介紹了，創建完帳戶後在“我的帳戶”-“API”中選擇“CREATE NEW API TOKEN”，然後下載 **kaggle.json** 檔。\n",
                "\n",
                "**建立 Kaggle 資料夾**："
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {
                "colab_type": "code",
                "id": "fffe6790",
                "outputId": "742be4a6-dc73-4424-b98a-0ab658f4cf12",
                "collapsed": False
            },
            "outputs": [],
            "source": [
                "!mkdir -p ~/.kaggle"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**將 kaggle.json 資料夾複製到指定資料夾**："
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {
                "colab_type": "code",
                "id": "672fdd1c",
                "outputId": "d49bde72-bf96-43cb-a368-11e2fbb6d36c",
                "collapsed": False
            },
            "outputs": [],
            "source": [
                "!cp /content/gdrive/MyDrive/kaggle.json ~/.kaggle/"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "**測試是否成功**："
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {
                "colab_type": "code",
                "id": "ff2b1c9c",
                "outputId": "f89397f0-0673-479b-8d04-80a41da6e22e",
                "collapsed": False
            },
            "outputs": [],
            "source": [
                "!kaggle competitions list"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "![image](https://github.com/11024244/mid/blob/main/jpg/04.png)\n",
                "\n",
                "**下載資料集**："
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {
                "colab_type": "code",
                "id": "fbb8d60f",
                "outputId": "13c01e3d-1827-4a4e-baa5-d9d2dc52c2f2",
                "collapsed": False
            },
            "outputs": [],
            "source": [
                "!kaggle datasets download -d paultimothymooney/kermany2018"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "![image](https://github.com/11024244/mid/blob/main/jpg/05.png)\n",
                "\n",
                "**將文件解壓至 Google 雲盤**："
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {
                "colab_type": "code",
                "id": "fc40f973",
                "outputId": "e4b42c7f-cd0d-4cb7-91f5-699bc7f7a8f7",
                "collapsed": False
            },
            "outputs": [],
            "source": [
                "!unzip \"/content/OCT2017.zip\" -d \"/content/gdrive/My Drive\""
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 數據讀取\n",
                "\n",
                "訓練、測試資料夾："
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {
                "colab_type": "code",
                "id": "27a95e70",
                "outputId": "77b32ba8-0c80-4186-b582-ea78945074ef",
                "collapsed": False
            },
            "outputs": [],
            "source": [
                "import os\n",
                "\n",
                "# 設定訓練和測試資料夾的路徑\n",
                "train_folder = os.path.join('/','content','gdrive','My Drive','OCT', 'train', '**', '*.jpeg')\n",
                "test_folder = os.path.join('/','content','gdrive','My Drive','OCT', 'test', '**', '*.jpeg')"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "這裡的 “**” 表示匹配所有子目錄。例如：\n",
                "\n",
                "```python\n",
                "Example:\n",
                "      If we had the following files on our filesystem:\n",
                "        - /path/to/dir/a.txt\n",
                "        - /path/to/dir/b.py\n",
                "        - /path/to/dir/c.py\n",
                "      If we pass \"/path/to/dir/*.py\" as the directory, the dataset would\n",
                "      produce:\n",
                "        - /path/to/dir/b.py\n",
                "        - /path/to/dir/c.py\n",
                "```\n"
            ]
        },
        {
            "cell_type": "markdown",
            "metadata": {},
            "source": [
                "## 數據處理\n"
            ]
        },
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {
                "colab_type
