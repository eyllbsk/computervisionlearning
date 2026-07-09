

import cv2
image = cv2.imread("images/sample.jpg")
print(type(image))

print(image)
print(image.shape)

cv2.imwrite("outputs/copy.jpg", image) 

print(image[0,0])
b = image[0,0][0]
g = image[0,0][1]
r = image [0,0][2]

print("Blue:", b)
print("Green:", g)
print("Red:" ,r)


image[0,0] = [0,0,255]
print(image[0,0])

cv2.imshow("Image", image)

cv2.waitKey(0)
cv2.destroyAllWindows() 