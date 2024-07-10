import cv2
from tkinter import Tk, filedialog

def drow_label(imge, results):
    text = "Medicine box"
    image = imge
    boxes = results[0].boxes.xyxy.tolist()
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        start_point = (int(x1), int(y2))
        start_point_text = (int(x1), int(y1))
        end_point = (int(x2), int(y1))
        new_end_point = (int(x2), int((y1 - 25)))
        color = (255, 0, 0)
        text_color = (255, 255, 1)
        thickness = 2
        fontFace = cv2.FONT_HERSHEY_DUPLEX
        fontScale = 1
        thickness = 2
        image = cv2.rectangle(image, start_point, end_point, color, thickness)
        image = cv2.rectangle(image, start_point_text, new_end_point, color, -1)
        image = cv2.putText(
            image,
            text,
            start_point_text,
            fontFace,
            fontScale,
            text_color,
            thickness,
            lineType=cv2.LINE_AA,
        )
    return image


def write_to_file(path, text):
    with open(path, "a") as file:
        file.write(text)


def choose_file():
    root = Tk()
    root.withdraw()  # يخفي النافذة الرئيسية
    file_path = filedialog.askopenfilename(
        title="اختر ملف"
    )  # يعرض نافذة اختيار الملفات
    return file_path


def crop_object(results, img):
    list_of_object = []
    boxes = results[0].boxes.xyxy.tolist()
    if len(boxes) == 0:
        list_of_object.append(img)
        return list_of_object
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cropped_image = img[int(y1) : int(y2), int(x1) : int(x2)]
        list_of_object.append(cropped_image)
    return list_of_object
