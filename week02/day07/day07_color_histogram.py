#görüntüde hangi renk daha baskın anlayabiliriz 

import cv2
import matplotlib.pyplot as plt

# Resmi oku
image = cv2.imread("images/sample.jpg")

if image is None:
    print("Resim yuklenemedi.")
    exit()

# OpenCV BGR kullandığı için RGB'ye çeviriyoruz
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Renk kanalları
colors = ("b", "g", "r")

plt.figure(figsize=(8,5))
plt.title("Color Histogram")
plt.xlabel("Pixel Value")
plt.ylabel("Pixel Count")

# Her renk kanalı için histogram hesapla
for i, color in enumerate(colors):

    histogram = cv2.calcHist(
        [image],
        [i],    #hangi kanalın histogramını çıkaracağımızı söylüyor 
        None,
        [256],
        [0,256]
    )

    plt.plot(histogram, color=color)      #her histogramı kendi renginden çiziyor 

plt.xlim([0,256])

plt.show()

cv2.imshow("Original", image)

cv2.waitKey(0)
cv2.destroyAllWindows()