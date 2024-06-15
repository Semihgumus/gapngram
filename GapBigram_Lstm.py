import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import LSTM, Dense, Embedding
from sklearn.metrics import classification_report, confusion_matrix
import tensorflow as tf 
from collections import Counter
import itertools
import seaborn as sns
import matplotlib.pyplot as plt

gpu_devices = tf.config.list_physical_devices('GPU')

if not gpu_devices:
    print("GPU kullanılamıyor. TensorFlow yalnızca CPU üzerinde çalışıyor.")
else:
    print("GPU kullanılabilir. TensorFlow GPU üzerinde çalışıyor.")

# Örnek veri setini oluştur
data = pd.read_csv("Hb_clear_filtered.csv")

# Score sütunundaki değerleri dönüştür
data = data[data['Score'] != 3]
data['Score'] = data['Score'].apply(lambda x: 0 if x in [1, 2] else 1)

# Hızlandırılmış bigramları oluşturacak fonksiyon
def create_bigrams(text):
    words = text.split()
    bigrams = [' '.join(pair) for pair in zip(words, words[1:])] + [' '.join((words[i], words[i + 2])) for i in range(len(words) - 2)]
    return bigrams

data['Bigrams'] = data['Text'].apply(create_bigrams)

# En çok geçen 10000 bigramı seç
all_bigrams = list(itertools.chain.from_iterable(data['Bigrams']))
most_common_bigrams = Counter(all_bigrams).most_common(500000)
most_common_bigrams1 = {bigram[0] for bigram in most_common_bigrams}

# Metinlerde sadece en çok geçen 10000 bigramı tut ve boş satırları sil
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

# Eğitim ve test setlerine ayır
X_train, X_test, y_train, y_test = train_test_split(X_padded, y, test_size=0.2, random_state=42)

# Modeli oluştur
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index)+1, output_dim=100, input_length=X_padded.shape[1]))
model.add(LSTM(units=100, dropout=0.2))
model.add(Dense(1, activation='sigmoid'))  # İkili sınıflandırma için sigmoid aktivasyonu

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', 'Precision', 'Recall'])

# Modeli eğit
model.fit(X_train, y_train, epochs=2, batch_size=64, validation_data=(X_test, y_test))

# Test verileri üzerinde tahmin yap
y_pred = model.predict(X_test)
y_pred_classes = (y_pred > 0.5).astype(int)
y_true = y_test

# Classification report'u görüntüle
print(classification_report(y_true, y_pred_classes))

# Confusion matrix'i hesapla
cm = confusion_matrix(y_true, y_pred_classes)

# Confusion matrix'i görselleştir (içindeki değerler olmadan)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=False, cmap="Blues", xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.show()
