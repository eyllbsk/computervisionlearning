import tensorflow as tf 
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
# applications--> hazır modeller, mobilenet_v2--> ait fonksiyonlar, preprocess_input--> mobile net'in istediği ön işleme fonksiyonu, resimleri mobile netin anlayacağı formata dönüştür 
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
# models --> model oluşturmak için kullanılan yapılar bulunur, Sequential --> katmanalar sırayla dizildiği model
from tensorflow.keras.layers import Resizing, GlobalAveragePooling2D,Dense
import numpy as np 

(x_train,y_train), (x_test,y_test) = cifar10.load_data()

print("Eğitim görüntüleri:", x_train.shape)
print("Eğitim etiketleri:", y_train.shape)
print("Test görüntüleri:", x_test.shape)
print("Test etiketleri:", y_test.shape)

x_train = preprocess_input(x_train.astype("float32"))
x_test = preprocess_input(x_test.astype("float32"))
# piksel tipini flaot32 yapar ve mobilenetin istediği formata dönüştürür 

base_model = MobileNetV2(
    weights = "imagenet",         # rastgele ağırlıklarla başlama, imagenet üzerinde öğrenilmiş ağırlıkları kullan 
    include_top = False,          # kendi dense katmanımızı eklicez, yani en son sınıflandırma katmanlarını kaldır 
    input_shape = (224,224,3)     # giriş görüntüsünün boyutunu söylüyor 
)

model = Sequential([
    Resizing(224,224), 
    base_model,                        # model sadece özellik çıkarıyor 
    GlobalAveragePooling2D(),
    Dense(128, activation = "relu"),     # mobile net'in çıkardığı özellikleri yorumlayacal 128 nöron oluşturuyor
    Dense(10, activation = "softmax")    # çıkış katmanı cifar10 10 sınıf içeriyor, softmax de olasılık dağılımını verir 
]) 

model.compile(                                    # artık eğitim başlayacak. sana nasıl öğreneceğini söylüyorum
    optimizer = "adam",                           # modelin öğrenme yöntemi           
    loss = "sparse_categorical_crossentropy",
    metrics = ["accuracy"]                        # modelin başarısını accuracy ile gösterir 
)

history = model.fit(                        # model eğitilir, eğitim boyunca oluşan bilgileri bize de verir, bunları da history değişkeninin içinde tutuyoruz 
    x_train,
    y_train,
    epochs = 5,
    batch_size = 32,                        # model her seferinde 32 resim alır, tahmin yapar, hatayı hesaplar ve ağırlıkları günceller
    validation_data = (x_test, y_test)
)

test_loss, test_accuracy = model.evaluate(x_test, y_test)                   # evaluate--> modeli test etmek, sadece test verisini çözüyor 

print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)


prediction = model.predict(x_test[:1])              # ilk elemandan başla, 1. elemana kadar al 
predicted_class = np.argmax(prediction)             # en büyük değerin indeksini bul 
print(predicted_class)


# Bu projede Transfer Learning kullanılarak MobileNetV2 tabanlı bir görüntü sınıflandırma modeli oluşturuldu.
# CIFAR-10 veri seti yüklendi ve MobileNetV2'nin beklediği formata dönüştürüldü.
# ImageNet üzerinde önceden eğitilmiş MobileNetV2, özellik çıkarıcı (Feature Extractor) olarak kullanıldı.
# Üzerine GlobalAveragePooling2D ve Dense katmanları eklenerek yeni sınıflandırıcı oluşturuldu.
# Model Adam optimizer ile eğitildi, test verisi üzerinde değerlendirildi ve son olarak yeni görüntüler için tahmin yapıldı.