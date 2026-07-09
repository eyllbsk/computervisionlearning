import cv2

# Resmi oku 
image = cv2.imread("images/sample.jpg")

# Resmin belirli bölgelerini yeniden renklendir
image[0:50,0:50] = [0,0,255]
image[50:100,50:100] = [0,255,0]
image[100:150,100:150] = [255,0,0]
image[20:40, 40:120] = [0,255,255]

# Resmi kes
cropped_image = image[40:140,60:160]

# Resmi yeniden boyutlandır
resized = cv2.resize(image, (500,300))

# Görüntüyü gri tona çevir 
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gri görüntüyü göster
cv2.imshow("Gray", gray)

# Boyutları yazdır
print(image.shape)
print(resized.shape)
print(gray.shape)

# Görüntüleri göster
cv2.imshow("Resized", resized)
cv2.imshow("Original", image)

# Kesilen kısmı göster
cv2.imshow("Crop", cropped_image)

cv2.waitKey(0)
cv2.destroyAllWindows()