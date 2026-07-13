# transfer learning; başka veri setinde eğitilmiş hazır bir modeli alıp, kendi problemimiz için yeniden kullanmaktır 
# normal CNN --> bütün ağırlıkları sıfırdan öğrenir.
# VGG16 ; basit ve anlaşılır , ResNet50; çok derin ağlarda başarı sağlar

# feature extraction; hazır model katmanlarını dondurup sadece eklediğin sınıflandırma kısmını eğitirsin 
# fine tuning; son birkaç katmanı açıp modeli kendi veri setine göre biraz daha uyarlarsın 
#Önce bu bilgileri kullanarak sadece yeni sınıflandırma katmanlarını eğitmek daha güvenlidir. Eğer veri setimiz büyükse ve daha 
# yüksek başarı hedefleniyorsa, son katmanlar açılarak Fine-Tuning uygulanabilir.

# tensorflow içinde hazır bulunan MobileNetv2 modelini projeye dahil ediyor, yani hazır modeli çağırıyoruz 
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense

base_model = MobileNetV2(
    weights = "imagenet",         #modeli rastgele başlatma, iamgenet üzerinde eğitilmiş ağırlıkları yükle , None olsaydı bütün ağırlıklar rastgele olurdu 
    include_top = False,          # son dense katmanlarını çıkar , geriye sadece özellik çıakran CNN kısmı kalsın 
    input_shape = (224,224,3)     # giriş resminin boyutunu belirliyor 224*224 RGB görüntü kabul eder
)

# hazır modelin ağırlıklarını dondur 
base_model.trainable = False 

print("MobileNetV2 başarıyla yüklendi")
print("Model eğtilebilir mi?", base_model.trainable)
print("Katman Sayısı:", len(base_model.layers))

# Yeni modeli oluştur 
model = Sequential([                    # katmanların birbiri ardına sıralandığı model, tüm katmanları köşeli parantez içine yaz 
    base_model,
    GlobalAveragePooling2D(),             # her filtrenin ortalamasını alır, örn son katmanda 2*2*3 çıktısı 3 filtre var -->[.., ..., ...] gibi bir çıktı olur 
    Dense(128, activation="relu"),      # 128 nöronlu tam bağlantılı katman oluştur . amaç; çıkarılan özellikleri yorumlamak 
    Dense(6, activation="softmax")      # çıkış katmanı. nedne 6? diyelim ki 6 sınıfımız var. softmax de çıktıyı olasılığa çeviriyor 
])

# model özetini yazdır 
model.summary()


# GlobalAveragePooling, Flatten'dan daha iyi. Flatten çok fazla parametre oluşturuyor . globalda daha az parametre ve overfitting daha az