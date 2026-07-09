# neden veri seti --> CNN kuralları örneklerden öğrenir
# MNIST --> el yazısı rakamlardan oluşur 

# Train Set --> öğrenme kısmı , model sadece train set'e bakarak öğrenir 
# Validation Set --> sınav kısmı, şuan iyi gidiliyor mu kontrolü yapılır
# Test Set --> model başarısı ölçülür

# Preprocessing --> resmi modele vermeden önce hazırlama işlemi 

# normalize; piksel değerlerini 0-255 aralığından 0-1 aralığına dönüştürerek modelin daha hızlı ve kararlı öğrenemesini sağlar
# one-hot encoding; kategorik etkiketleri, her sınıfın yalnızca kendi konumunda 1 bulunan ve diğer tüm değerleri 0 olan vektörlere dönüştürme işlemi
# CNN tüm girişlerin aynı boyutta olmasını bekler, bu yüzden resimler aynı boyutta olmalı 

model.compile(
    optimizer = "adam",                  # modelin yaptığı hatalara göre ağırlıkları güncelleyen algoritma
    loss = "categorical_crossentropy",   # model ne kadar hata yapıyor, amaç loss'u 0'a yaklaştırmak
    metrics = ["accuracy"]               # başarı oranını göster
)

# categorical_crossentropy --> one hot decoding  sparse --> one hot yoksa kullan


model.fit(
    x_train,            #eğitim görüntüleri 
    y_train,            #görüntülerin doğru cevapları,etiketler
    epochs = 10,        # 1 epoch = modelin eğitim veri setinin tamamını bir kez görmesi ---> fazla olursa model ezberlemeye başlar(overfitting)
    batch_size = 32,    # modele tek seferde verdiğimiz grup büyüklüğü 1000 resmi 10 kere 100 şeklinde vermek gibi 
    validation_data = (x_val, y_val)  
)

# forward propagation --> tahmin 
# loss hesaplanır --> backpropagation --> weight update --> sonraki batch 