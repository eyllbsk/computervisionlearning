from ultralytics import YOLO            # ultralytics kütüphanesinin içindeki YOLO sınıfını programa dahil ediyor, hazır YOLO modellerini kullanmamızı sağlar 
import cv2                              # görüntü okuma, gösterme ve temel görüntü işleme işlemleri için kullanacağız 
import os 

print(os.getcwd())

model = YOLO("yolov8n.pt")              #bu dosya önceden milyonlarca görüntü ile eğitilmiş model. modeli biz eğitmiyoruz, hazır kullanıyoruz
image = cv2.imread(r"C:\Users\eylul\OneDrive\Masaüstü\computervisionlearning\week04\day16\day16_YOLO_project\example.jpg")

if image is None:
    print("Resim okunamadı")
    exit()


results = model(image, conf=0.50)                  # görüntüyü YOLO modeline gönderir ve nesne tespiti sonuçlarını döndürür
print("Result değişkeni:" ,(results))               # YOLO'nun görüntü için döndürdüğü bütün sonuç yapısını gösterir (tespit kutularo, sınıflar,güven skoru, görüntü boyutu, çalışma süreleri)

print("\nResult veri tipi:", type(results))        # her görüntü için ayrı bir sonuç oluşturur ve bunları liste içinde tutar , [0]--> listedeki ilk görüntünün sonucunu seçer

#ilk görüntünün tespit kutularını al 
boxes = results[0].boxes
person_count = 0

for box in boxes: 
    class_id = int(box.cls[0])
    confidence = float(box.conf[0])
    coordinates = box.xyxy[0]
    
    class_name = model.names[class_id]

    if class_name == "person":
        person_count += 1

    print("toplam insan sayısı:", person_count)
    
    print("Nesne:", class_name)
    print("Güven skoru:%", round(confidence*100,2))
    print("Koordinatlar:", coordinates)
    print("-----------------------")

#Tespşt kutularını terminalde gösterir 
print("\nBounding box bilgileri:", (boxes))


annotated = results[0].plot()           # YOLO'nun bulduğu kutular resmin üzerine çizilir
annotated = cv2.resize(annotated, (900,600))

cv2.imshow("YOLO Detection", annotated)     # resmi gösterme 

cv2.waitKey(0)
cv2.destroyAllWindows()


# boxes.cls; sınıf numaraları, boxes.conf; güven skorları , boxes.xyxy; bounding box koordinatları, model.names; sınıf numarasını sınıf adına çevirir