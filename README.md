# Аннотация
Целью этого проекта является подготовка данных для обучения и само обучение архитектур нейронных сетей (YOLOv4 darknet, Faster-RCNN Detectron2) на предмет распознавания конусов на гоночной трассе.  

Сам датасет и координаты Bounding Box к нему доступны в репозитории MIT: https://github.com/cv-core/MIT-Driverless-CV-TrainingInfra/tree/master/CVC-YOLOv3

# Обучение нейросети YOLOv4 

FromCSV_to_TXTdarknet.py
 
Скрипт для перевода координат BB из файла all.csv в .txt файлы с для каждого .jpg изображения.

Содержание текстового файла:

    <object-class> <x_center> <y_center> <width> <height>

Где:
 
    <object-class> = метка класса. Класс в данном коде один: 'Cone'
 
    <x_center> = ((<absolute_x> + <absolute_width>)/2)/ <image_width>
 
    <y_center> = ((<absolute_y> + <absolute_height>)/2)/ <image_height>
 
    <height> = <absolute_height> / <image_height>
 
    <width> = <absolute_width> / <image_width>
  
Пример:

  1 0.716797 0.395833 0.216406 0.147222
  0 0.687109 0.379167 0.255469 0.158333
  1 0.420312 0.395833 0.140625 0.166667

Yolov4_grayscale_images_train.ipynb
  
Подготовка к обучению модели по туториалу https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects и само обучение на GPU Google Colab. 
  
