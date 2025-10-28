import torch
import cv2
import numpy as np

# Betöltjük a YOLO modellt
model = torch.hub.load('WongKinYiu/yolov7', 'custom', path='yolov7.pt', force_reload=True)

# Kép betöltése
image_path = 'path/to/your/image.jpg'  # Cseréld le a saját képed elérési útjára
img = cv2.imread(image_path)

# Kép előfeldolgozása
img_resized = cv2.resize(img, (640, 640))
img_normalized = img_resized / 255.0  # Normalizálás
img_input = np.expand_dims(img_normalized, axis=0)  # Dimenziók bővítése

# Előrejelzés készítése
results = model(img_input)

# Eredmények feldolgozása
for result in results.xyxy[0]:  # Eredmények
    x1, y1, x2, y2, conf, cls = result.numpy()  # Koordináták, konfidencia, osztály
    if int(cls) == 0:  # Csak a repülőgépeket (osztály 0)
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(img, f'Airplane: {conf:.2f}', (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

# Eredmények megjelenítése
cv2.imshow("Detected Airplanes", img)
cv2.waitKey(0)
cv2.destroyAllWindows()