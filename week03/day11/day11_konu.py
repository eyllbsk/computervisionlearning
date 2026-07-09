# CNN ; Convolutional Neural Network
# özellikleri kendisi öğrenir, filtreler eğitim sırasında öğrenilir

# resim --> convolution(kenarları bul) --> activation(ReLU)(negatifleri sil) --> pooling(boyutları küçült) --> convolution(daha karmaşık şekilleri öğren) --> ReLU --> pooling
# --> flatten --> fully connected layer --> output

# convolution ; resmin üzerinde küçük bir pencere dolaştırır. pencere--> filter, kernel. her geçtiği yerde hesap yapar,
# sonra yeni bir görüntü oluşturur --> feature map (önemli bilgileri gösteren yeni resim)

# padding; convolution yaparken görüntü küçülür, resmin etrafına 0'lar ekliyoruz. --> kenarlardaki bilgiler kaybolmaz, daha fazla özellik öğrenilebilir

# Activation Function; gereksiz bilgileri siliyor, ağı daha hızlı eğitiyor. 
# pooling ; resim boyutu küçülür, hesaplama azalır. overfitting azalır, önemli bilgiler korunur  

# flatten; feature map tek boyuta dönüştürür. çünkü dense kavramı sadece vektör kabul eder. yani 2D görüntü --> 1D liste 

# Fully Connected Layer (dense layer); bütün bilgileri kullanarak bunun hangi sınıf olduğuna karar verir. 
# --> her nöron farklı bir nörona bağlı 

# output layer; artık sınıflar var her nöron bir sınıfı temsil eder

# softmax; son katmandaki değerleri olasılığa dönüştüren aktivasyon fonksiyonu. çıktılar toplamı 1 olur 