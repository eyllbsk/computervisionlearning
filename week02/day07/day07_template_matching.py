import cv2

# Büyük resmi oku
image = cv2.imread("images/sample.jpg")

# Aranacak küçük resmi oku
template = cv2.imread("images/template.jpg")

if image is None or template is None:
    print("Resim yuklenemedi.")
    exit()

# Gri tona çevir. çünkü template matching renk yerien desen benzerliği üzerinden daha iyi çalışır 
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Template Matching uygula
result = cv2.matchTemplate(      #opencv küçük resmi, büyük resmin üzerinde tek tek gezdiriyor 
    gray_image,
    gray_template,
    cv2.TM_CCOEFF_NORMED    #1-->mükemmel eşleşme  0-->benzemiyor -1-->çok kötü eşleşme 
)

# En iyi eşleşmeyi bul. minMaxLoc() ile en yüksek benzerlik değerine sahip konum tespit edildi
min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)   

print("Benzerlik Orani:", max_val)

# Template'in boyutları
h, w = gray_template.shape

# Eşleşen bölgenin başlangıç noktası
top_left = max_loc

# Sağ alt köşe. templatein genişliği ve yüksekliği eklenerek sağl alt koşe bulundu
bottom_right = (top_left[0] + w, top_left[1] + h)

# Dikdörtgen çiz
cv2.rectangle(     
    image,
    top_left,
    bottom_right,
    (0, 255, 0),
    2
)

cv2.imshow("Original", image)
cv2.imshow("Template", template)

cv2.waitKey(0)
cv2.destroyAllWindows()

#bu yöntem sabit boyut ve sabit yönelimli nesnelerin tespitinde etkili olup, 
# ölçek ve döndürme değişimlerine karşı hassastır 