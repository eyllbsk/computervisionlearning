import cv2
import numpy as np 

#Resmi oku 
image = cv2.imread("images/sample.jpg")
if image is None: 
    print("resim yüklenemedi")
    exit()

height, width = image.shape[:2]

#Resmin merkezini bul 
center = (width // 2, height // 2)

#Döndürme matrisini oluştur, nerenin etrafında döndüreceğim, kaç derece,ölçek 
matrix = cv2.getRotationMatrix2D(center,-30,1)
rotated_30 =cv2.warpAffine(image,matrix, (width,height))

#resimleri döndürme
rotated_90 = cv2.rotate(image,cv2.ROTATE_90_CLOCKWISE)


flip_left_right = cv2.flip(image,1)
flip_up_down = cv2.flip(image,0)
flip_both = cv2.flip(image,-1)


cv2.imshow("rotated30", rotated_30)
cv2.imshow("rotated90", rotated_90)
cv2.imshow("horizontal", flip_left_right)
cv2.imshow("vertical",flip_up_down)
cv2.imshow("both", flip_both)



cv2.waitKey(0)
cv2.destroyAllWindows()