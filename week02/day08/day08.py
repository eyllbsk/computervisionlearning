#haar cascade --> opencv nin hazır olarak sunduğu klasik bir nesne tespit algoritması 
#renklerle çalışmaz ilgilendiği aydınlık-karanlık geçişleri 

import cv2

# Haar Cascade modelini yükle
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

# Resmi oku
image = cv2.imread("images/face.jpg")

if image is None:
    print("Resim yuklenemedi.")
    exit()

# Gri tona çevir
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Yüzleri bul
faces = face_cascade.detectMultiScale(        #resmin tamamını tarıyor
    gray,
    scaleFactor=1.1,   #resmi farklı boyutlarda tarıyor, yüz küçük veya büyük olabilir 
    minNeighbors=5     #yanlış tespitleri azaltır 
)

print("Bulunan yuz sayisi:", len(faces))

# Kutu çiz
for (x, y, w, h) in faces:

    cv2.rectangle(
        image,
        (x, y),
        (x+w, y+h),
        (0,255,0),
        2
    )

cv2.imshow("Face Detection", image)

cv2.waitKey(0)
cv2.destroyAllWindows()


'''Haar Cascade yöntemi kullanılarak görüntüdeki yüzler tespit edildi. 
Görüntü önce gri tona çevrildi, ardından detectMultiScale() fonksiyonu ile 
farklı ölçeklerde taranarak yüzlerin koordinatları belirlendi. 
Tespit edilen her yüzün etrafına cv2.rectangle() fonksiyonu ile yeşil kutular çizdirildi.'''