
# Обучение нейросети Faster - RCNN
Источники:
1) Detectron2 Docs https://detectron2.readthedocs.io/en/latest/index.html#
2) Roboflow article https://blog.roboflow.com/how-to-train-detectron2/

Convert_from_txtDarknet_to_json_format.ipynb

Перевод координатBonding Box из формата 

YOLO format 

  1 0.716797 0.395833 0.216406 0.147222
  
  0 0.687109 0.379167 0.255469 0.158333
  
  1 0.420312 0.395833 0.140625 0.166667

в Coco Json Format

![alt text](https://miro.medium.com/max/444/1*wleaRUAKGGwxe3YMcKWRbA.png)

Faster_R-CNN_Detectron2.ipynb

Обучение модели и тестирование на полученных в ходе обучения весах.
