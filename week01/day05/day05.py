#canny edge detection 

import cv2

image = cv2.imread("images/sample.jpg")

if image is None:
    print("resim yüklenemedi")
    exit()

#Canny tek kanallı görüntü ister
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

blur = cv2.GaussianBlur(gray, (5,5), 0)

#iki eşik kullanıyoruz. ilki lower threshold, bunun altındaysa kenar değil 
#upper threshold bunun üstündeyse kesin kenar 
#arasındaysa yakınında güçlü bir kenar varsa kabul ediyor, yoksa siliyor 
edges = cv2.Canny(blur,100,200)

#eşik değeri altı siyah üstü beyaz. ret --> kullanılan threshold değeri, binary --> oluşturulan yeni görüntü
ret, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

#tam tersi arabanın çoğu siyah kalanlar beyaz olucak
ret, binary_inv = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)

cv2.imshow("Binary Inverse", binary_inv)

#127 üstündeki değerleri 127ye sabitliyor 
ret, trunc = cv2.threshold(gray, 127, 255, cv2.THRESH_TRUNC)

cv2.imshow("Truncate", trunc)

#127nin altını sıfır yap, üstünü olduğu gibi bırak
ret, tozero = cv2.threshold(gray, 127, 255, cv2.THRESH_TOZERO)

cv2.imshow("ToZero", tozero)


#otsu threshold en uygun thresholdu hesaplıyor
ret, otsu = cv2.threshold(
    gray,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU    #binary threshold uygula ama değerini sen hesapla
)
cv2.imshow("Otsu", otsu)
print(ret)

#ret-->otsunun bulduğu threshold 

#sobel de tek kanal ister, 64 bit float kullanıyoruz hesaplama yaparken +/- değerler üretebilir
# 1--> x yönünde türev al 0--> y yönünde türev alma
sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1)

#sobel sonucunu ekranda gösterilebilir normal görüntüye dönüştürür
sobel_x = cv2.convertScaleAbs(sobel_x)
sobel_y = cv2.convertScaleAbs(sobel_y)




cv2.imshow("Original", image)
cv2.imshow("Gray", gray)
cv2.imshow("Blur", blur)
cv2.imshow("Canny", edges)
cv2.imshow("sobel x", sobel_x)
cv2.imshow("sobel y", sobel_y)
cv2.imshow("Binary", binary)

cv2.waitKey(0)
cv2.destroyAllWindows()


#canny'den önce gaussian yoksa canny bunları kenar sanabilir. canny noislerı kenar sanabilir
#canny renklerle ilgilenmez. onun için parlaklık (intensity) değiişimi önemli 

''' Grayscale                 canny'nin içinde zaten sobel var 
      │
Gaussian Blur
      │
Sobel (Gradient Hesabı)
      │
Non-Maximum Suppression
      │
Double Threshold
      │
Hysteresis
      │
Canny Sonucu  '''