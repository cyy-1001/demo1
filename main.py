from scipy.stats import anderson_ksamp
from ultralytics import YOLO
#modelini başlatma
#model = YOLO('yolov9e.yaml')  # 'n' model boyutunu temsil eder, 'n', 's', 'm', 'l', 'x' olabilir

# Modeli eğitme
#model.train(data='data.yaml', epochs=100, imgsz=320)

model = YOLO(r'C:\Users\Berat\PycharmProjects\Yolov8\runs\detect\train\weights\best.pt')

# Doğrulama işlemi
results = model.val(data='data.yaml')



