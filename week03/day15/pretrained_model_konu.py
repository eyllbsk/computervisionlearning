import tensorflow as tf          #tensorflow adını kısaltıp tf olarak kullanıcam demek 
from tensorflow.keras.datasets import cifar10  
#tf--> ana kütüphane, keras--> ts'nin derin öğrenme kısmı, datasets--> hazır veri setlerinin bulunduğu bölüm, cifar10--> 10 sınıflı hazır görüntü veri seti  

(x_train,y_train), (x_test,y_test) = cifar10.load_data()
# modeli önce eğit, sonra teste sokuyoruz. cifar10 veri setini bilgisayara indirir 

print("Eğitim görüntüleri:", x_train.shape)
# bütün veri seti gelir (50000,32,32,3)
print(x_train[0].shape)
# ilk resim (32,32,3)
# x_train[0][0].shape --> ilk satır (32,3)     x_train[0][0][0].shape --> ilk piksel (3),

x_train = tf.image.resize(x_train, (224,224))
x_test = tf.image.rezise(x_test, (224,224))
# reshape()--> resmin içeriği değişmiyor, verinin görünüşü değişiyor , resize()--> gerçekten resim değişiyor yani 32*32 den 224*224 olursa yeni pikseller ekleniyor 
# mobileNet 222*224 ister ikisinin de boyutu aynı olmalı , ama 50000 görüntünün hepsini tekte büyütmek çok fazla RAM, batch batch büyütmek daha güvenli olur 

print(x_train.shape)
print(x_test.shape)