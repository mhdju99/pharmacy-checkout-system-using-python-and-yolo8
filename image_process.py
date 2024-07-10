
import cv2



def resize_img(image, per):
    down_width = image.shape[0] - (image.shape[0] * per)
    down_height = image.shape[1] - (image.shape[1] * per)
    down_points = (int(down_height), int(down_width))
    resized_down = cv2.resize(image, down_points,
                               interpolation=cv2.INTER_LINEAR)
    return resized_down
