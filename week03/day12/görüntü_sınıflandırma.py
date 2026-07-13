from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.datasets import mnist

import numpy as np 

(x_train, y_train), (x_test, y_test) = mnist.load_data()
# x_train; eğitim görüntüleri , y_train; etiketler, x_test; modelin daha önce görmediği test görüntüsü, y_test; cevapları 
# mnis.load_data() --> eğitim ve test veri setlerini ayrı döndürür

print(x_train.shape)
print(y_train.shape)

print(x_test.shape)
print(y_test.shape)

x_train = x_train / 255.0     #normalization, artık tüm görüntüler 0-1 arasında
x_test = x_test / 255.0       # neden 255? görüntüler 8 bit için piksel değerleri 0-1 aralığına gelir

x_train = x_train.reshape(-1,28,28,1)       # 1--> kanal (mnist;siyah beyaz)  -1--> python'a ilk boyutu (kaç örnek olduğunu) sen hesapla diyoruz
x_test = x_test.reshape(-1,28,28,1)         # neden reshape --> CNN'nin beklediği (yükseklik, genişlik, kanal) formatına dönüştürür

print("Normalize ve reshape sonrası:")
print(x_train.shape)
print(x_test.shape)

# Model oluştur 
model = Sequential()

# 1. convolution + pooling 
# 32 tane filtre 3*3 boyutunda, relu; negatif değerleri sıfır yapar 
model.add(Conv2D(32, (3,3), activation="relu", input_shape = (28,28,1)))

#2*2 alanlardan en büyük değeri alır, görüntü boyutu küçültür,öğrenme yapmaz 
model.add(MaxPooling2D(pool_size = (2,2)))


# Tek boyutlu hale getir. (13*13*32) boyutundaki feature map'i çevirir 
model.add(Flatten())

# Karar verme katmanı, 128 tane karar veren nöron 5408*128 + 128 (bias) = 692,352  her dense katmanındaki nöronun 1 biası vardır 
model.add(Dense(128, activation = "relu"))

#Çıkış katmanı: 10 rakam olduğu için 10 nöron 128*10+10 = 1290
model.add(Dense(10,activation="softmax"))

model.summary()

# parametre = (giriş * çıkış) + çıkış 

model.compile(
    optimizer = "adam",
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]
)

print(y_train[:10])

history = model.fit(         #history;  eğitim sonunda loss, accuracy, val_loss, val_accuracy saklıyor
    x_train,
    y_train,
    epochs = 5,
    batch_size = 32,
    validation_split = 0.2    #eğitim görüntüsünün %20'si validation oluyor   60000--> 48000 train--> 12000 validation 48000/32 = 1500 batch
)

#Test verisi üzerinde modeli değerlendir 
test_loss, test_accuracy = model.evaluate(x_test,y_test)

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
 
prediction = model.predict(x_test[:1])
print(prediction)


# MNIST veri seti üzerinde basit bir CNN modeli kurdum, veriyi normalize edip CNN formatına getirdim, modeli eğittim, test verisinde değerlendirdim ve tek görüntü için tahmin aldım 
