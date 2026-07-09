import cv2
import numpy as np

# Resmi oku
image = cv2.imread("images/bluedetection.jpg")

if image is None:
    print("Resim yuklenemedi.")
    exit()

# BGR'den HSV'ye cevir
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Mavi renk araligi, alt ve üst sınır belirler 
#[hue,saturation,value]  s ve v opencv'de 0-255 arası değişir 
lower_blue = np.array([90, 50, 50])
upper_blue = np.array([130, 255, 255])

# Maske olustur, bu aralıktaki pikselleri beyaz(seçilen renk bulundu) yapar, kalanlarını siyah(seçilen renk değil) yapar 
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Maskeyi orijinal resme uygula, sadece maskede beyaz kalan yerleri orijinal görüntüde gösterir 
result = cv2.bitwise_and(image, image, mask=mask)

# Maskedeki kontürleri bul. maskedeki beyaz bölgeleri buluyor 
contours, hierarchy = cv2.findContours(
    mask,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# En büyük kontürü bul
if len(contours) > 0:

    largest_contour = max(contours, key=cv2.contourArea)

    area = cv2.contourArea(largest_contour)

    print("Alan:", area)

    # Çok küçük alanları alma
    if area > 500:

        x, y, w, h = cv2.boundingRect(largest_contour)   #kontürü çevreleyen en küçük dikdörtgeni hesaplıyor 

        cv2.rectangle(       #mavi nesnenin etrafına yeşil kutu çiziliyor 
            image,
            (x, y),
            (x+w, y+h),
            (0,255,0),
            2
        )

cv2.imshow("Original", image)
cv2.imshow("Mask", mask)
cv2.imshow("Result", result)

cv2.waitKey(0)
cv2.destroyAllWindows()

#ışık değişse bile hue çok değişmez bu yüzden renk seçmek daha kolaydır. hue = renk tonu,
#saturation = doygunluk , value = parlaklık 

'''rüntü HSV renk uzayına çevrildi. Belirlenen mavi renk aralığı cv2.inRange() 
ile maske olarak çıkarıldı. Ardından cv2.bitwise_and() kullanılarak yalnızca mavi 
renge sahip bölgeler görüntü üzerinde bırakıldı.'''


'''HSV renk uzayında oluşturulan maske üzerinde kontür analizi yapıldı. En büyük kontür 
seçilerek alanı hesaplandı ve belirlenen eşik değerinin üzerindeki nesnenin etrafına 
cv2.boundingRect() ile dikdörtgen çizildi. Böylece yalnızca istenen renkteki nesnenin 
konumu tespit edilmiş oldu.'''