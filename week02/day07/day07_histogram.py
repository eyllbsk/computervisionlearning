#histogram --> görüntüdeki piksel yoğunluklarının (parlaklık veya renk değerlerininin) 
#kaç tane olduğunu gösteren grafik 

import cv2
import matplotlib.pyplot as plt 

#resmi oku 
image = cv2.imread("images/sample.jpg")

if image is None:
    print("Resim yuklenemedi.")
    exit()

#gri tona çevir 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#Histogram hesapla 
histogram = cv2.calcHist(
    [gray],  #görüntü 
    [0],     #kanal (0=gri)
    None,    #maske yok yani tğm görüntünün histogramı çıkarılıyor 
    [256],   #256 kutu 
    [0,256]   #piksel aralığı 
)

#görüntüyü göster 
cv2.imshow("gray image", gray)

#histogram çiz 
plt.figure(figsize=(8,4))
plt.title("Grayscale Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Pixel Count")

plt.plot(histogram)

plt.xlim([0,256])

plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

'''Gri tonlu görüntünün histogramı incelendiğinde piksel yoğunluğunun büyük kısmının 
orta parlaklık seviyelerinde toplandığı görüldü. Siyah ve beyaz uç değerlerde daha az piksel bulunduğu 
için görüntünün ne aşırı karanlık ne de aşırı parlak olduğu söylenebilir. Histogram, görüntünün parlaklık 
dağılımını analiz etmek için kullanıldı.'''