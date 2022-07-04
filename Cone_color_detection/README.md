Код, который классицирует на фотографиях конуса по цвету , для дальнейшего обучения датасета из конусов на нескольких классах.

3 классc:
1) Желтый конус (с чёрным посередине)
2) Оранжевый и синий конуса (с белым посередине)
3) Неизвестный конус (большой оранжевый конус или который не удалось чётко классифицировать)  


![загрузка](https://user-images.githubusercontent.com/71214107/157867803-00b3b83e-35c1-4bf5-95b3-2ab447e43ce4.png)


Используемые библиотеки: OpenCV, Pillow, NumPY, Pandas

Планируется:
1) Написать код по аугментации данных и пересчёта координат ограничивающих рамок.
Сделано: 

1.1) ![Horizontal flip](https://github.com/SerSurguchev/ConeDetection/blob/main/Cone_color_detection/data_augmentation.py#L12)

1.2) ![Image Scaling](https://github.com/SerSurguchev/ConeDetection/blob/main/Cone_color_detection/data_augmentation.py#L44)

# Результаты 
![vid_37_frame_255](https://user-images.githubusercontent.com/71214107/177032350-b062fae3-ac8e-44e1-bb8d-92d0af469343.jpg)
![vid_38_frame_797](https://user-images.githubusercontent.com/71214107/177032353-f3a96aa4-942a-4692-a173-2902cce7f91e.jpg)
![vid_79_frame_87](https://user-images.githubusercontent.com/71214107/177032354-525e4ccb-a158-40c8-ad3e-cd402f9659f3.jpg)
![vid_5_frame_2549](https://user-images.githubusercontent.com/71214107/177032358-6326bdcc-6697-4925-935c-983ddac467c0.jpg)
![vid_18_frame_2265](https://user-images.githubusercontent.com/71214107/177032361-6ddb40f2-b418-4691-b708-fe2aaebe0897.jpg)
![vid_28_frame_2590](https://user-images.githubusercontent.com/71214107/177032362-c89244ff-0a80-46f9-b9e8-a0f896716940.jpg)
![vid_31_frame_2147](https://user-images.githubusercontent.com/71214107/177032363-18c1905f-4235-4d7e-8e89-a3a52cb52004.jpg)
![vid_31_frame_2153](https://user-images.githubusercontent.com/71214107/177032364-a5bb1ad9-a77f-4800-ad85-7c09e0779885.jpg)
