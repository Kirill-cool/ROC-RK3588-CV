import cv2

# Пробуем физическое устройство камеры
cap = cv2.VideoCapture("/dev/video12", cv2.CAP_V4L2)

if not cap.isOpened():
    print("Не удалось открыть камеру через V4L2.")
else:
    print("Камера успешно открыта через V4L2!")
    ret, frame = cap.read()
    if ret:
        cv2.imshow("V4L2 Test", frame)
        cv2.waitKey(0)
    cap.release()
    cv2.destroyAllWindows()
