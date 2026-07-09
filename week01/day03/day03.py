import cv2

#Resmi oku
image = cv2.imread("images/sample.jpg")

if image is None:
    print("resim yüklenemedi")
    exit()

#Resmi döndür 
rotated90=cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
rotated180 = cv2.rotate(image , cv2.ROTATE_180)
rotated270 = cv2.rotate(image , cv2.ROTATE_90_COUNTERCLOCKWISE)

#Boyutları yazdır 
print("Original Shape:",image.shape)
print("Rotated Shape: " , rotated90.shape)

#Görüntüleri göster 
cv2.imshow ("Original" , image)
cv2.imshow ("Rotated" , rotated90)
cv2.imshow ("180", rotated180)
cv2.imshow ("270", rotated270)

cv2.waitKey(0)
cv2.destroyAllWindows() 


#Boyutları al
height, width = image.shape[:2]

#Resmin merkezini hesapla
center = width // 2, height // 2

#45 derece döndürme matrisi oluştur 
matrix = cv2.getRotationMatrix2D(center, -45, 1)

#Döndürme işlemini uygula 
rotated = cv2.warpAffine(image,matrix, (width,height))

cv2.imshow("original", image)
cv2.imshow("45 degree rotation", rotated)

cv2.waitKey(0)
cv2.destroyAllWindows()

#Flip işlemleri
flip_left_right = cv2.flip(image,1)
flip_up_down = cv2.flip(image,0)
flip_both = cv2.flip(image,-1)

#Sonuçları göster
cv2.imshow("original", image)
cv2.imshow("horizontal", flip_left_right)
cv2.imshow("vertical",flip_up_down)
cv2.imshow("both", flip_both)

cv2.waitKey(0)
cv2.destroyAllWindows()
