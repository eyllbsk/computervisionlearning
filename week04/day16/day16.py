# YOLO -->  you only look once , tek seferde nesneyi bul hem sınıflandır 
# resim alınır --> küçük karelere ayrılır --> her parça kendine sorar benim içimde nesne var mı? varsa kutu koordinatı, güven skoru, sınıf --> tüm parçaların sonuçları birleşir
# tek seferde hem sınıfı hem konum tahmin eder 

# Backbone; resimdeki önemli featureları çıkarır, özellik çıkaran CNN de diyebiliriz. feature maps çıktı, sistem sadece önemli bilgileri taşıyor
# Neck; farklı ölçeklerdeki özellikleri birleştirmek, güncel YOLO'larda FPN(feature pyramid network) ve PAN (path aggregation network) kullanılır
# Head; tahmin yapar 

# giriş görüntüsü --> resize(hepsi aynı boyutta olmalı) --> backbone --> neck --> head--> NMS (non maximum suppression), aynı nesne için birden fazla kutu oluşabilir
# en yüksek güven skoruna sahip kutu seçilir

# YOLOv8;  anchor free yapı , daha hızlı eğitim, daha yüksek doğruluk, küçük nesnelerde daha başarılı, gerçek zamanlı çalışabilir, çoklu görev desteği
# bunlar sayesinde günümüzde en popüler sürümlerden biri  