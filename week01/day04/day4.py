import cv2

#Resmi oku 
image = cv2.imread("images/sample.jpg")

if image is None: 
    print("Resim yüklenemedi")
    exit()

#gaussian blur
gaussian = cv2.GaussianBlur(image, (5, 5),0)

#median blur 
median = cv2.medianBlur(image, 5)

#bilateral blur 
bileteral = cv2.bilateralFilter(image, 9, 75, 75)

cv2.imshow("original", image)
cv2.imshow("gaussian blur", gaussian)
cv2.imshow("median blur", median)
cv2.imshow("bileteral filter", bileteral)


cv2.waitKey(0)
cv2.destroyAllWindows()




# Day 04 - Image Filtering

## Öğrendiklerim
'''
- Blur görüntüyü yumuşatır.
- Gürültüyü azaltmak için kullanılır.
- Kernel büyüdükçe blur artar.
- Gaussian Blur genel amaçlıdır.
- Median Blur Salt & Pepper Noise için uygundur.
- Bilateral Filter kenarları koruyarak blur uygular. '''