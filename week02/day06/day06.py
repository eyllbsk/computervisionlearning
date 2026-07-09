import cv2

# Resmi oku
image = cv2.imread("images/sample.jpg")

if image is None:
    print("Resim yüklenemedi.")
    exit()

# Kontür genelde direkt renkli resimde bulunmaz,
# önce görüntüyü sadeleştiririz.
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Gaussian Blur ile küçük gürültüleri azaltırız.
blur = cv2.GaussianBlur(gray, (5, 5), 0)

# Canny ile kenarları tespit ederiz.
edges = cv2.Canny(blur, 100, 200)

# Kontür bulma
contours, hierarchy = cv2.findContours(
    edges,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

# edges --> kontür bulacağımız siyah beyaz görüntü
# cv2.RETR_EXTERNAL --> sadece dış kontürleri al
# cv2.CHAIN_APPROX_SIMPLE --> gereksiz noktaları azalt

print("Bulunan kontur sayısı:", len(contours))

# Orijinal resmin kopyasını al
contour_image = image.copy()

# Kontürleri tek tek gez
for i, contour in enumerate(contours):

    # Kontürün kapladığı alanı hesaplar
    area = cv2.contourArea(contour)

    # Küçük alanlı kontürler genelde gürültü veya gereksiz detaydır.
    # Bu yüzden sadece alanı 20'den büyük olanları alıyoruz.
    if area > 20:

        # Kontürün çevresini hesaplar
        perimeter = cv2.arcLength(contour, True)

        # Kontürü daha az noktayla yaklaşık olarak ifade eder.
        # Böylece köşe sayısını bulabiliriz. cv2.approxPolyDP() ile yaklaşık köşe sayıları hesaplandı.
        #araba görüntüsü karmaşık olduğu içim koşe sayıları net bir şekil vermedi ama kontür yapısını analiz etmeye yardım etti 
        approx = cv2.approxPolyDP(
            contour,
            0.02 * perimeter,
            True
        )

        # Yaklaşık kontürün köşe sayısı
        corner_count = len(approx)

        x, y, w, h = cv2.boundingRect(contour)    #kontürün etrafını saran dikdörtgenin bilgilerini verir 

        cv2.rectangle(        #görüntüyü dikdörtgene çizer, mavi dikdörtgen her kontürü tamamen içine alacak şekilde oluşturulan bounding box 
            contour_image, 
            (x,y),
            (x + w, y + h), 
            (255,0,0),
            2
        )
        
        #yazıyı ekrana yaz 
        cv2.putText(
            contour_image, 
            f"Kose: {corner_count}",
            (x, y - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 255),
            2 
            )    
        #(x, y-10) --> yazının kutunun biraz üstüne gelmesini sağlar
        #(0,0,255) --> kırmızı  0.5 --> yazı boyutu 2 -->kalınlığı 

        

        

        print(f"{i+1}. kontur")
        print("Alan:", area)
        print("Cevre:", perimeter)
        print("Kose sayisi:", corner_count)
        print("----------------")

        # Kontürleri çiz
        cv2.drawContours(contour_image, [contour], -1, (0, 255, 0), 2)

        # -1 --> tüm kontürü çiz
        # (0,255,0) --> yeşil
        # 2 --> çizgi kalınlığı

cv2.imshow("Original", image)
cv2.imshow("Edges", edges)
cv2.imshow("Contours", contour_image)

cv2.waitKey(0)
cv2.destroyAllWindows()

'''Kontürler tespit edildikten sonra cv2.boundingRect() fonksiyonu ile her kontürün 
etrafına onu tamamen kapsayan en küçük dikdörtgen çizdirildi. Araba üzerinde cam, far 
ve tekerlek gibi farklı bölgeler ayrı kontür olarak algılandığı için birden fazla Bounding Box oluştu.
 Bu yöntem, nesne tespiti ve görüntü işleme uygulamalarında nesnelerin konumunu belirlemek için yaygın olarak kullanılmaktadır.'''