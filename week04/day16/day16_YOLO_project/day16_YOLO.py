from ultralytics import YOLO            # ultralytics kütüphanesinin içindeki YOLO sınıfını programa dahil ediyor, hazır YOLO modellerini kullanmamızı sağlar 
import cv2                              # görüntü okuma, gösterme ve temel görüntü işleme işlemleri için kullanacağız 
import os 

print(os.getcwd())

model = YOLO("yolov8n.pt")              #bu dosya önceden milyonlarca görüntü ile eğitilmiş model. modeli biz eğitmiyoruz, hazır kullanıyoruz
image = cv2.imread(r"C:\Users\eylul\OneDrive\Masaüstü\computervisionlearning\week04\day17_YOLO_project\example.jpg")

if image is None:
    print("Resim okunamadı")
    exit()


results = model(image, conf=0.60)                  # görüntüyü YOLO modeline gönderir ve nesne tespiti sonuçlarını döndürür
annotated = results[0].plot()           # YOLO'nun bulduğu kutular resmin üzerine çizilir
annotated = cv2.resize(annotated, (900,600))

cv2.imshow("YOLO Detection", annotated)     # resmi gösterme 
cv2.waitKey(0)
cv2.destroyAllWindows()
