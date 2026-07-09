import cv2
import numpy as np

# Kamerayi ac
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Kamera acilamadi.")
    exit()

# Nesnenin gecmis merkez noktalarini tutar
points = []

while True:     #sonsuz döngü başlattık, kameradan sürekli görüntü alacağız 

    # Kameradan frame oku
    ret, frame = cap.read()

    if not ret:
        print("Frame okunamadi.")
        break

    # BGR goruntuyu HSV renk uzayina cevir
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Mavi renk araligi
    lower_blue = np.array([90, 50, 50])
    upper_blue = np.array([130, 255, 255])

    # Mavi renk icin maske olustur
    mask = cv2.inRange(hsv, lower_blue, upper_blue)

    # Maskedeki gurultuleri azalt
    mask = cv2.erode(mask, None, iterations=1)
    mask = cv2.dilate(mask, None, iterations=2)

    # Maskedeki konturleri bul
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) > 0:

        # En buyuk konturu sec
        largest_contour = max(contours, key=cv2.contourArea)

        # Konturun alanini hesapla
        area = cv2.contourArea(largest_contour)

        # Kucuk gurultuleri ele
        if area > 500:

            # Konturu icine alan en kucuk cemberi hesapla
            (x, y), radius = cv2.minEnclosingCircle(largest_contour)

            center = (int(x), int(y))
            radius = int(radius)

            # Cember cok kucukse cizme
            if radius > 10:

                # Topun etrafina yesil cember ciz
                cv2.circle(
                    frame,
                    center,
                    radius,
                    (0, 255, 0),
                    2
                )

                # Merkez noktasini kirmizi nokta ile goster
                cv2.circle(
                    frame,
                    center,
                    5,
                    (0, 0, 255),
                    -1   #içi dolu daire demek 
                )

                # Merkez noktasini listeye ekle
                points.append(center)

                # Sadece son 50 noktayi tut, böylece ekran tamamen çizgiyle dolmaz 
                if len(points) > 50:
                    points.pop(0)

                # Hareket izini ciz
                for i in range(1, len(points)):
                    cv2.line(
                        frame,
                        points[i - 1],
                        points[i],
                        (255, 0, 255),
                        2
                    )

                # Ekrana bilgi yaz
                cv2.putText(
                    frame,
                    "Blue Ball",
                    (center[0] - 40, center[1] - radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 0),
                    2
                )

                cv2.putText(
                    frame,
                    f"Center: {center}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"Radius: {radius}",
                    (20, 60),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2
                )

    # Pencereleri goster
    cv2.imshow("Ball Tracking", frame)
    cv2.imshow("Mask", mask)

    # q tusuna basinca cik
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kamerayi kapat
cap.release()
cv2.destroyAllWindows()


'''Bu mini projede kameradan alınan görüntü HSV renk uzayına çevrildi. 
Mavi renk aralığına göre maske oluşturuldu. Maske üzerinde gürültü azaltma 
işlemleri uygulandı. Ardından kontürler bulunarak en büyük mavi bölge seçildi. 
Bu bölgeyi içine alan en küçük çember minEnclosingCircle() ile hesaplandı. 
Topun merkezi, yarıçapı ve hareket izi gerçek zamanlı olarak görüntü üzerinde 
gösterildi.'''