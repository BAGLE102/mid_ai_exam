# google-colab平台訓練模型案例

南華大學 人工智慧期中報告 
11024102 况旻諭

# 數據集介紹

![image](https://github.com/11024244/mid/blob/main/jpg/01.png)

數據集來自**Kaggle**，品質很高，由知名醫院的專業人員嚴格審核標註，如圖所示數據有4種類別：

 •**CNV**：具有新生血管膜和相關視網膜下液的脈絡膜新血管形成
 
 •**DME**：糖尿病性黃斑水腫與視網膜增厚相關的視網膜內液
 
 •**DRUSEN**：早期AMD中存在多個玻璃疣
 
 •**NORMAL**：視網膜正常，沒有任何視網膜液或水腫
 
![image](https://github.com/11024244/mid/blob/main/jpg/02.png)

檔大小約為5GB，8萬多張圖像，分為訓練，測試，驗證三個資料夾，每個資料夾按照種類不同分成4個子資料夾，其次是具體圖像檔。

# 數據集下載

**掛載資料夾**：

```py
from google.colab import drive

# 掛載 Google Drive，以便讀取和存儲數據
drive.mount('/content/gdrive/')
```
按照提示進行驗證，結果如下：

![image](https://github.com/11024244/mid/blob/main/jpg/03.png)

**kaggle資料下載**：

創建**kaggle**帳戶並下載**kaggle.json**檔。 創建帳戶這裡就不介紹了，創建完帳戶後在“我的帳戶”-“API”中選擇“CREATE NEW API TOKEN”，然後下載**kaggle.json**檔。

**建立kaggle資料夾**：
```py
!mkdir -p ~/.kaggle
```
**將kaggle.json資料夾複製到指定資料夾**：
```py
!cp /content/gdrive/MyDrive/kaggle.json ~/.kaggle/
```
**測試是否成功**：
```py
!kaggle competitions list
```
![image](https://github.com/11024244/mid/blob/main/jpg/04.png)

**下載資料集**：
```py
!kaggle datasets download -d paultimothymooney/kermany2018
```
![image](https://github.com/11024244/mid/blob/main/jpg/05.png)

**將文件解壓至google雲盤**：
```py
!unzip "/content/OCT2017.zip" -d "/content/gdrive/My Drive"
```
![image](https://github.com/11024244/mid/blob/main/jpg/06.png)

# 數據讀取

指定訓練和測試資料夾路徑：
```py
import os

import os

# 設定訓練和測試資料夾的路徑
train_folder = os.path.join('/','content','gdrive','My Drive','OCT', 'train', '**', '*.jpeg')
test_folder = os.path.join('/','content','gdrive','My Drive','OCT', 'test', '**', '*.jpeg')
```
資料夾內圖像以 **/*.jpeg 形式讀取，** 代表子資料夾內所有 .jpeg 文件。

以下為舉例:
```py
Example:
      If we had the following files on our filesystem:
        - /path/to/dir/a.txt
        - /path/to/dir/b.py
        - /path/to/dir/c.py
      If we pass "/path/to/dir/*.py" as the directory, the dataset would
      produce:
        - /path/to/dir/b.py
        - /path/to/dir/c.py
```
# 數據處理
為模型定義數據處理函數，包括圖像解碼、縮放、標籤解析等：
```py
def input_fn(file_pattern, labels,
             image_size=(224, 224),
             shuffle=False,
             batch_size=64, 
             num_epochs=None, 
             buffer_size=4096,
             prefetch_buffer_size=None):

    # 建立標籤映射表
    table = tf.contrib.lookup.index_table_from_tensor(mapping=tf.constant(labels))
    num_classes = len(labels)

    def _map_func(filename):
        # 從檔名中提取標籤
        label = tf.strings.split([filename], delimiter=os.sep).values[-2]
        
        # 讀取並解碼圖像
        image = tf.image.decode_jpeg(tf.io.read_file(filename), channels=3)
        image = tf.image.convert_image_dtype(image, dtype=tf.float32)
        
        # 調整圖像大小以符合 VGG16 模型的輸入形狀
        image = tf.image.resize(image, size=image_size)
        
        # 回傳圖像和其對應的 one-hot 編碼標籤
        return (image, tf.one_hot(table.lookup(label), num_classes))
    
    # 創建資料集
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=shuffle)
    
    # 處理資料集的隨機化和重複
    if num_epochs is not None and shuffle:
        dataset = dataset.apply(tf.data.experimental.shuffle_and_repeat(buffer_size, num_epochs))
    elif shuffle:
        dataset = dataset.shuffle(buffer_size)
    elif num_epochs is not None:
        dataset = dataset.repeat(num_epochs)
    
    # 使用並行處理來加速數據處理
    dataset = dataset.apply(
        tf.data.experimental.map_and_batch(map_func=_map_func,
                                           batch_size=batch_size,
                                           num_parallel_calls=os.cpu_count()))
    
    # 預取數據以提高效率
    dataset = dataset.prefetch(buffer_size=prefetch_buffer_size)
    
    return dataset

```
# 模型訓練
```py
import tensorflow as tf
import os

# 設置日誌顯示等級
tf.logging.set_verbosity(tf.logging.INFO)

# 數據集標籤
labels = ['CNV', 'DME', 'DRUSEN', 'NORMAL']

# 載入 VGG16 模型，不包括最後 3 個全連接層
keras_vgg16 = tf.keras.applications.VGG16(input_shape=(224, 224, 3),
                                          include_top=False)

# 將 VGG16 的輸出展平
output = keras_vgg16.output
output = tf.keras.layers.Flatten()(output)

# 添加一個全連接層，並使用 softmax 激活函數進行多類別分類
predictions = tf.keras.layers.Dense(len(labels), activation='softmax')(output)

# 定義模型
model = tf.keras.Model(inputs=keras_vgg16.input, outputs=predictions)

# 冻结 VGG16 模型的最後四層
for layer in keras_vgg16.layers[:-4]:
    layer.trainable = False

# 定義優化器
optimizer = tf.train.AdamOptimizer()

# 編譯模型，使用 categorical_crossentropy 作為損失函數
model.compile(loss='categorical_crossentropy', 
              optimizer=optimizer,
              metrics=['accuracy'])

# 設定訓練日誌的配置
est_config = tf.estimator.RunConfig(log_step_count_steps=10)

# 將 Keras 模型轉換為 Estimator
estimator = tf.keras.estimator.model_to_estimator(model, model_dir='/content/gdrive/My Drive/estlogs', config=est_config)

BATCH_SIZE = 32  # 設定批次大小
EPOCHS = 2  # 設定訓練輪次

# 訓練模型
estimator.train(input_fn=lambda: input_fn(test_folder,
                                         labels,
                                         shuffle=True,
                                         batch_size=BATCH_SIZE,
                                         buffer_size=2048,
                                         num_epochs=EPOCHS,
                                         prefetch_buffer_size=4))

```
