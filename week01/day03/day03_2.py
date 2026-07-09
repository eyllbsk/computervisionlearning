import cv2
import numpy as np 
 
#resmi oku
image = cv2.imread("images/sample.jpg")

#sağa 80, aşağı 40 piksel kaydır 
tx = 80
ty = 40

matrix = np.float32([
    [1,0,tx],
    [0,1,ty]
])

translated = cv2.warpAffine(
    image,
    matrix,
    (image.shape[1], image.shape[0])
)

cv2.imshow("original", image)
cv2.imshow("translated", translated)

cv2.waitKey(0)
cv2.destroyAllWindows