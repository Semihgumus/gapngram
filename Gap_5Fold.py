import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from sklearn.metrics import classification_report
import tensorflow as tf
from collections import Counter
import itertools
from keras import backend as K

# GPU bellek ayarlarını scriptin başında yap
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        # Sanal cihaz yapılandırmasını belirle
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=4096)])  # Bellek sınırını ayarla (örneğin 4GB)
        print("GPU kullanılabilir. TensorFlow GPU üzerinde çalışıyor.")
    except RuntimeError as e:
        print(e)
else:
    print("GPU kullanılamıyor. TensorFlow yalnızca CPU üzerinde çalışıyor.")

# Örnek veri setini oluştur
data = pd.read_csv("Hb_clear_filtered.csv")

# Score sütunundaki değerleri dönüştür
data = data[data['Score'] != 3]
data['Score'] = data['Score'].apply(lambda x: 0 if x in [1, 2] else 1)

# Hızlandırılmış bigramları oluşturacak fonksiyon
"""def create_bigrams(text):
    words = text.split()
    bigrams = [' '.join(pair) for pair in zip(words, words[1:])] + [' '.join((words[i], words[i + 2])) for i in range(len(words) - 2)]
    return bigrams"""
def create_bigrams(text):
    words = text.split()
    return [' '.join(words[i:i + 2]) for i in range(len(words) - 1)]
data['Bigrams'] = data['Text'].apply(create_bigrams)

# En çok geçen 750000 bigramı seç
all_bigrams = list(itertools.chain.from_iterable(data['Bigrams']))
most_common_bigrams = Counter(all_bigrams).most_common(500000)
most_common_bigrams1 = {bigram[0] for bigram in most_common_bigrams}

# Metinlerde sadece en çok geçen bigramları tut ve boş satırları sil
def filter_bigrams(text):
    return [bigram for bigram in text if bigram in most_common_bigrams1]

# Bigramları filtrele
data['Filtered_Bigrams'] = data['Bigrams'].apply(filter_bigrams)
data = data[data['Filtered_Bigrams'].astype(bool)]  # Boş satırları kaldır

# Metinleri ve etiketleri ayır
X = data['Filtered_Bigrams']
y = data['Score']

# Metinleri tokenize et ve pad_sequences ile aynı uzunlukta yap
tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
X_tokenized = tokenizer.texts_to_sequences(X)
X_padded = pad_sequences(X_tokenized)

# Modeli oluşturma fonksiyonu
def create_model(input_length, vocab_size):
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size, output_dim=100, input_length=input_length))
    model.add(LSTM(units=100, dropout=0.2))
    model.add(Dense(1, activation='sigmoid'))  # İkili sınıflandırma için sigmoid aktivasyonu
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', tf.keras.metrics.Precision(), tf.keras.metrics.Recall()])
    return model

# Çapraz doğrulama
kf = KFold(n_splits=5, shuffle=True, random_state=42)
fold = 1
reports = []

for train_index, test_index in kf.split(X_padded):
    print(f'Fold {fold}')
    
    X_train, X_test = X_padded[train_index], X_padded[test_index]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    
    model = create_model(input_length=X_padded.shape[1], vocab_size=len(tokenizer.word_index) + 1)
    
    # Model eğitimi sırasında bellek kullanımını kontrol etmek için tf.data API'yi kullanarak veri kümelerini optimize edin
    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(64).prefetch(tf.data.AUTOTUNE)
    test_dataset = tf.data.Dataset.from_tensor_slices((X_test, y_test)).batch(64).prefetch(tf.data.AUTOTUNE)
    
    model.fit(train_dataset, epochs=2, validation_data=test_dataset)
    
    y_pred = model.predict(X_test)
    y_pred_classes = (y_pred > 0.5).astype(int)
    report = classification_report(y_test, y_pred_classes, output_dict=True)
    reports.append(report)
    print(classification_report(y_test, y_pred_classes))
    fold += 1
    
    # Bellek temizleme
    K.clear_session()
    del model
# Sonuçların ortalamasını hesapla
avg_report = {}
for key in reports[0].keys():
    avg_report[key] = {}
    for metric in reports[0][key].keys():
        metric_values = [report[key][metric] for report in reports if isinstance(report[key][metric], (int, float))]
        avg_report[key][metric] = np.mean(metric_values)

print("Ortalama Sonuçlar:")
print(avg_report)


