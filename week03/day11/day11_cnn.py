from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

# Model oluştur 
model = Sequential()

# 1.convolution + pooling 
model.add(Conv2D(32, (3,3), activation="relu", input_shape = (224,224,3)))
model.add(MaxPooling2D(pool_size = (2,2)))

#2. convolution + pooling 
model.add(Conv2D(64, (3,3), activation="relu" ))
model.add(MaxPooling2D (pool_size = (2,2) ))

# Feature Map'i tek boyutlu hale getir 
model.add(Flatten())

# Karar verme katmanı 
model.add(Dense(128, activation="relu"))
 
# Çıkış katmanı 
model.add(Dense(6, activation= "softmax"))

# Model özetini yazdır
model.summary()
