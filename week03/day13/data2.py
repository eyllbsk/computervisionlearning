from tensorflow.keras.preprocessing.image import ImageDataGenerator     # verileri eğitim sırasında otomatik değiştirir 
from tensorflow.keras.datasets import mnist         
import matplotlib.pyplot as plt                                       
 # resimleri ekranda göstermek için kullanıyoruz 

# veri setini yükle 
(x_train, y_train), (x_test,y_test) = mnist.load_data()

# normalize 
x_train = x_train / 255.0

# kanal ekle 
x_train = x_train.reshape(-1,28,28,1)
 
# image data generator oluşturma 
datagen = ImageDataGenerator(
    rotation_range = 15,        # resmi 15 derece sağa sola döndürebilir
    zoom_range = 0.2,           # %20 yakınlaştırabilir veya uzaklaştırabilir 
    width_shift_range = 0.1,    # yatay kaydırma 
    height_shift_range = 0.1    # dikey kaydırma 
)

# ilk resmi aldı 
image = x_train[0]
image = image.reshape((1,28,28,1))
# image generator tek resim istemez (resim sayısı, yükseklik, genişlik, kanal)

# yeni görüntü üretme, her çağırıldığında yeni bir görüntü oluşur
iterator = datagen.flow(image,batch_size=1)
#batchsize = 1; her seferinde 1 tane artırılmış görüntü üret 

# yeni bir pencere oluşturuyor. genişlik = 8, yükseklik = 6
plt.figure(figsize=(8,6))   

for i in range(6):

    augmented = next(iterator)[0]      # her döngüde bir görüntü üret 
    
    plt.subplot(2,3,i+1)               # 2 satır, 3 sütun yani 6 kutu oluştu. neden i+1 matplotlib 1'den başlıyor i olsaydı ilk değer 0 olurdu 
    plt.imshow(augmented.squeeze(), cmap="gray")
    plt.axis("off")                    # kenar çizgilerini kaldır 

plt.show()