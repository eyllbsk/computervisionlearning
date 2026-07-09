import cv2
import numpy as np  #renk aralıklarını dizi olarak tanımlamak için kullanılır 

# Kamerayi ac. 0 genelde varsayılan kameradır 
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera acilamadi.")
    exit()

#nesnenin merkez noktalarını tutacak liste 
points = []

while True:
    # Kameradan bir frame oku. ret -->görüntünün okunup okunmadığını söyler. 
    #frame --> kameradan gelen görüntü 
    ret, frame = cap.read()

    if not ret:
        print("Frame okunamadi.")
        break

    # BGR'den HSV'ye cevir. renk seçmek hsv de daha kolay #H: 90-130   → mavi tonları 
    # S: 50-255   → yeterince canlı renkler V: 50-255   → yeterince parlak renkler
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mavi renk araligi
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Maske olustur. mavi olan yerleri beyaz yapar, diğerleri siyah yapar 
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Maskedeki gurultuleri azalt
    mask = cv2.erode(mask, None, iterations=1)    #maskede oluşan küçük gürültüleri azaltır, beyaz bölgeleri küçültür 
    mask = cv2.dilate(mask, None, iterations=2)   #beyaz bölgeleri tekrar büyütür, kopuk yerleri birleştirmeye yardım eder

    # Konturleri bul. maskede beyaz bölgelerin sınırlarını bulur, mavi nesnenin kontürlerini çıkarır 
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:

        # En buyuk konturu sec
        largest_contour = max(contours, key=cv2.contourArea)

        # Alanini hesapla
        area = cv2.contourArea(largest_contour)

        # Kucuk nesneleri alma
        if area > 500:

            # Bounding Box
            x, y, w, h = cv2.boundingRect(largest_contour)

            # Yesil kutu ciz
            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # Merkez noktasi
            center_x = x + w // 2
            center_y = y + h // 2

            #merkez noktaasını listeye ekle 
            points.append((center_x, center_y))

            #son 50 noktayı tut 
            if len(points) > 50:
                points.pop(0)

            # Merkezi kirmizi nokta ile goster
            cv2.circle(
                frame,
                (center_x, center_y),
                5,
                (0, 0, 255),
                -1
            )

            # Hareket izini ciz
            for i in range(1, len(points)):

                cv2.line(
                    frame,
                    points[i - 1],
                    points[i],
                    (255, 0, 255),
                    2
                )


            # Nesne ismi
            cv2.putText(
                frame,
                "Blue Object",
                (x, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2
            )

            # Alan bilgisini yaz
            cv2.putText(
                frame,
                f"Area: {int(area)}",
                (x, y + h + 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 255),
                2
            )

            # Merkez koordinatini yaz
            cv2.putText(
                frame,
                f"Center: ({center_x}, {center_y})",
                (x, y + h + 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 0),
                2
            )

            # Terminale yazdir
            print(f"Alan: {int(area)} | Merkez: ({center_x}, {center_y})")

    cv2.imshow("Frame", frame)
    cv2.imshow("Mask", mask)

    # q tusuna basinca cik
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

'''Kamera görüntüsü HSV renk uzayına çevrildi. Mavi renk aralığı belirlenerek maske oluşturuldu.
 Maske üzerindeki gürültüler erosion ve dilation ile azaltıldı. Daha sonra kontürler bulunarak 
 en büyük mavi bölge seçildi. Bu bölgenin etrafına kutu çizildi, merkezi hesaplandı ve 
 görüntü üzerinde gösterildi.'''