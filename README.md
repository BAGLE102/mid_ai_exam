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
2. 下載數據集
在 Kaggle 中創建 API Token 並下載 kaggle.json 文件。將 kaggle.json 移至 .kaggle 目錄：

bash
複製程式碼
!mkdir -p ~/.kaggle
!cp /content/gdrive/My\ Drive/kaggle.json ~/.kaggle/
!kaggle datasets download -d paultimothymooney/kermany2018
!unzip "/content/kermany2018.zip" -d "/content/gdrive/My Drive/OCT"
3. 數據處理
設定訓練與測試數據夾路徑
python
複製程式碼
import os

train_folder = os.path.join('/','content','gdrive','My Drive','OCT', 'train', '**', '*.jpeg')
test_folder = os.path.join('/','content','gdrive','My Drive','OCT', 'test', '**', '*.jpeg')
定義數據處理函數
python
複製程式碼
def input_fn(file_pattern, labels, image_size=(224,224), shuffle=False, batch_size=64, num_epochs=None, buffer_size=4096, prefetch_buffer_size=None):
    # 數據讀取與處理
    # 詳細實現...
4. 模型訓練
構建 VGG16 模型
使用 VGG16 作為特徵提取器，並添加分類層：

python
複製程式碼
import tensorflow as tf

keras_vgg16 = tf.keras.applications.VGG16(input_shape=(224,224,3), include_top=False)
output = tf.keras.layers.Flatten()(keras_vgg16.output)
predictions = tf.keras.layers.Dense(len(labels), activation='softmax')(output)
model = tf.keras.Model(inputs=keras_vgg16.input, outputs=predictions)
模型編譯與訓練
鎖定 VGG16 的前置層以保留預訓練權重，使用 Adam 優化器和分類交叉熵損失。

python
複製程式碼
for layer in keras_vgg16.layers[:-4]:
    layer.trainable = False

optimizer = tf.train.AdamOptimizer()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

estimator = tf.keras.estimator.model_to_estimator(model, model_dir='/content/gdrive/My Drive/estlogs')
estimator.train(input_fn=lambda:input_fn(test_folder, labels, shuffle=True, batch_size=32, buffer_size=2048, num_epochs=2, prefetch_buffer_size=4))
結語
本專案演示了如何使用 Google Colab 和 TensorFlow 來完成視網膜圖像分類的深度學習模型訓練，對視網膜疾病的分類自動化提供了一個良好的實踐範例。

go
複製程式碼

將此內容複製到 `README.md` 文件中，即可作為該專案的 README！





你說：
全部寫成一個markdown
ChatGPT 說：
ChatGPT
以下是整合後的完整 Markdown 文件，適合直接放入 README.md 中。

markdown
複製程式碼
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

```python
from google.colab import drive
drive.mount('/content/gdrive/')
2. 下載數據集
在 Kaggle 中創建 API Token 並下載 kaggle.json 文件。然後，將 kaggle.json 移至 .kaggle 目錄，並使用 Kaggle API 下載數據集：

bash
複製程式碼
!mkdir -p ~/.kaggle
!cp /content/gdrive/My\ Drive/kaggle.json ~/.kaggle/
!kaggle datasets download -d paultimothymooney/kermany2018
!unzip "/content/kermany2018.zip" -d "/content/gdrive/My Drive/OCT"
3. 數據處理
設定訓練與測試數據夾路徑
指定訓練及測試數據文件的路徑，以便後續加載數據。

python
複製程式碼
import os

train_folder = os.path.join('/','content','gdrive','My Drive','OCT', 'train', '**', '*.jpeg')
test_folder = os.path.join('/','content','gdrive','My Drive','OCT', 'test', '**', '*.jpeg')
定義數據處理函數
使用以下函數處理數據，包括圖像加載、標籤解析、圖像縮放等操作：

python
複製程式碼
def input_fn(file_pattern, labels, image_size=(224,224), shuffle=False, batch_size=64, num_epochs=None, buffer_size=4096, prefetch_buffer_size=None):
    table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(labels))
    num_classes = len(labels)

    def _map_func(filename):
        label = tf.string_split([filename], delimiter=os.sep).values[-2]
        image = tf.image.decode_jpeg(tf.read_file(filename), channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        image = tf.image.resize_images(image, size=image_size)
        return (image, tf.one_hot(table.lookup(label), num_classes))

    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)

    if num_epochs is not None and shuffle:
        dataset = dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size, num_epochs))
    elif shuffle:
        dataset = dataset.shuffle(buffer_size)
    elif num_epochs is not None:
        dataset = dataset.repeat(num_epochs)

    dataset = dataset.apply(
        tf.contrib.data.map_and_batch(map_func=_map_func,
                                      batch_size=batch_size,
                                      num_parallel_calls=os.cpu_count()))
    
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
    
    return dataset
4. 模型訓練
構建 VGG16 模型
使用 VGG16 作為特徵提取器，並添加分類層：

python
複製程式碼
import tensorflow as tf

# 加載 VGG16 模型
keras_vgg16 = tf.keras.applications.VGG16(input_shape=(224,224,3), include_top=False)
output = tf.keras.layers.Flatten()(keras_vgg16.output)
predictions = tf.keras.layers.Dense(len(labels), activation='softmax')(output)

# 定義模型
model = tf.keras.Model(inputs=keras_vgg16.input, outputs=predictions)
模型編譯與訓練
鎖定 VGG16 的前置層以保留預訓練權重，使用 Adam 優化器和分類交叉熵損失來訓練模型。

python
複製程式碼
# 設置前置層為不可訓練
for layer in keras_vgg16.layers[:-4]:
    layer.trainable = False

optimizer = tf.train.AdamOptimizer()
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])

# 使用 estimator API 進行訓練
estimator = tf.keras.estimator.model_to_estimator(model, model_dir='/content/gdrive/My Drive/estlogs')
estimator.train(input_fn=lambda:input_fn(test_folder, labels, shuffle=True, batch_size=32, buffer_size=2048, num_epochs=2, prefetch_buffer_size=4))
結語
本專案演示了如何使用 Google Colab 和 TensorFlow 來完成視網膜圖像分類的深度學習模型訓練，對視網膜疾病的分類自動化提供了一個良好的實踐範例。
