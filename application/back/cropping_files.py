from ultralytics import YOLO
from PIL import Image
import io
import os
import math
import numpy as np

model_path = os.path.abspath(os.path.join("model_repository", "models", "1", "best.pt"))
upload_dir = os.path.abspath("upload")

# Создаём директорию upload, если она не существует
os.makedirs(upload_dir, exist_ok=True)

def crop_image(image_data):
    best_confidence = -1  # Начальное значение для хранения лучшей вероятности
    cropped_image = None  # Переменная для хранения лучшего обрезанного изображения
    image = Image.open(io.BytesIO(image_data))
    image = image.convert("RGB")

    # Генерируем уникальное имя файла для хранения
    image_path = os.path.join(upload_dir, "uploaded_image.jpg")
    image.save(image_path, format="JPEG")

    model = YOLO(model_path)
    results = model(image)

    cropped_images = []
    for result in results:
        if hasattr(result, 'boxes'):
            total_boxes = len(result.boxes)
            
            for i, box in enumerate(result.boxes):
                # Проверяем, что box имеет атрибут xyxy для извлечения координат
                if hasattr(box, 'xyxy') and box.xyxy.shape[-1] == 4:
                    x_min, y_min, x_max, y_max = box.xyxy[0].tolist()

                    # Попытка получить значение вероятности
                    confidence = None
                    if hasattr(box, 'confidence'):
                        confidence = box.confidence.item() * 100
                    elif hasattr(box, 'score'):
                        confidence = box.score.item() * 100
                    elif hasattr(box, 'conf'):
                        confidence = box.conf.item() * 100

                    # Проверяем, что значение confidence не NaN и больше текущего лучшего значения
                    if confidence is not None and not math.isnan(confidence):

                        if confidence > best_confidence:
                            best_confidence = confidence

                            # Расширяем рамки с учетом padding и обрезаем изображение
                            padding = 0.1  # Процент увеличения рамки
                            height, width = image.size
                            x_min = max(0, int(x_min - padding * (x_max - x_min)))
                            y_min = max(0, int(y_min - padding * (y_max - y_min)))
                            x_max = min(width, int(x_max + padding * (x_max - x_min)))
                            y_max = min(height, int(y_max + padding * (y_max - y_min)))

                            # Обрезаем изображение по расширенным рамкам и сохраняем как лучшее
                            cropped_image = image.crop((x_min, y_min, x_max, y_max))
                            best_cropped_image = np.array(cropped_image)
                    else:
                        return f"Ошибка: Probability для Bounding Box {i + 1} является NaN или отсутствует."
            else:
                return "Ошибка: result не содержит атрибут boxes."

        if best_cropped_image is not None:
            best_cropped_image_pil = Image.fromarray(best_cropped_image)
            best_cropped_image_pil.save(os.path.join(upload_dir, "original.png"))

            for rotation in range(1, 4):
                rotated_image = np.rot90(best_cropped_image, rotation)
                rotated_image_pil = Image.fromarray(rotated_image)
                rotated_image_pil.save(os.path.join(upload_dir, f"rotated_{rotation * 90}.png"))

            os.remove(image_path)

            return np.array(best_cropped_image_pil)
            
        else:
            return "Нет подходящего изображения с уверенностью выше порога."

    