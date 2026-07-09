from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

#model oluştur 
model = Sequential()

#ilk convolution katmanı
model.add(Conv2D(
    filters = 32,
    kernel_size = (3,3),
    activation="relu"
))

print("İlk Convolution Katmanı eklendi")

# 1. pooling katmanı 
model.add(MaxPooling2D(pool_size=(2,2)))

#2. convolution katmanı 
model.add(
    Conv2D(
        filters=64, 
        kernel_size=(3,3),
        activation="relu"
    )
)
print("iki convolution ve bir pooling katmanı başarıyla eklendi")

# ilk katman görüntüdeki temel özellikleri (kenar,çizgi,köşe gibi) öğrenir. İkinci katmanda ise daha karmaşık özellikler (şekiller, nesne parçaları gibi) öğrenileceği için filtre sayısı artırılarak modelin daha fazla özellik öğrenmesi sağlanır 


