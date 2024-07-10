from image_process import *
from object_dedact import *
from sift import *
import cv2
from util import *
from ultralytics import YOLO

from prettytable import PrettyTable


def main():
    prices = {
        "Bmatrix": 7500,
        "Sinuline": 11500,
        "Skelflex": 15500,
        "Superfix": 9500,
        "Pasperan": 6500,
        "Laxine": 5500,
        "Fadierin": 20500,
        "Esostom": 17500,
        "Esemprozol": 16500,
        "Deopflam": 14500,
        "Defetal": 11500,
        "stopLerg": 18500,
        "Arthatoks": 19500,
    }

    model = YOLO("best.pt")
    folder_path = "product _database/"
    
    sift_features = extract_sift_features_from_folder(folder_path)
    while True:
        tabel = PrettyTable(["Item", "price"])
        print("ready")

        image_path = choose_file()
        orignal = cv2.imread(image_path)
        img = resize_img(orignal, 0.8)
        result = prdict(model, img)
        ims = drow_label(img, result)
        img1 = resize_img(ims, 0.3)
        cv2.imshow("Clustered Image", img1)
        list_of_object = crop_object(result, img)
        total = 0
        for object in list_of_object:
            key, des = extract_sift_features_from_image(object)
            best_match = match_sift_features(key, des, sift_features)
            name = best_match["image_path"]
            price = prices[best_match["image_path"]]
            if best_match:
                tabel.add_row([name, f"{price}" + " SP"])

                total = total + price
            else:
                print("No match found.")
        write_to_file("invoic.txt", f"{tabel}\n")
        write_to_file("invoic.txt", f"total is............. = {total}" + " SP")
        print(f"{tabel}\n")
        print(f"total is............. = {total}" + " SP")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
