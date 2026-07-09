import cv2
import matplotlib.pyplot as plt

# Resmi oku
image = cv2.imread("images/sample.jpg")

if image is None:
    print("Resim yuklenemedi.")
    exit()

# Gri tona çevir
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Histogram Equalization uygula
equalized = cv2.equalizeHist(gray)   
#histogram analiz ediliyor , piksel değerleri yeniden dağıtıyor, kontrastı artırıyor

# Orijinal histogram
hist_original = cv2.calcHist(
    [gray],
    [0],
    None,
    [256],
    [0,256]
)

# Equalized histogram
hist_equalized = cv2.calcHist(
    [equalized],
    [0],
    None,
    [256],
    [0,256]
)

# Görüntüleri göster
cv2.imshow("Original Gray", gray)
cv2.imshow("Equalized", equalized)

# Histogramları çiz
plt.figure(figsize=(10,4))

plt.subplot(1,2,1)
plt.title("Original Histogram")
plt.plot(hist_original)
plt.xlim([0,256])

plt.subplot(1,2,2)
plt.title("Equalized Histogram")
plt.plot(hist_equalized)
plt.xlim([0,256])

plt.tight_layout()
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()

'''Histogram Equalization yöntemi ile görüntünün kontrastı artırıldı. 
cv2.equalizeHist() fonksiyonu kullanılarak piksel değerleri tüm parlaklık 
aralığına daha dengeli dağıtıldı. Böylece görüntüdeki detaylar daha belirgin 
hale geldi ve histogram daha geniş bir yoğunluk dağılımı gösterdi.'''