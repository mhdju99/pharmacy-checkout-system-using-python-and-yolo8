import cv2
import os


def extract_sift_features_from_folder(folder_path):
    features_list = []
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    for index, image_file in enumerate(image_files):
        image_path = os.path.join(folder_path, image_file)
        img = cv2.imread(image_path)
        keypoints, descriptors=  extract_sift_features_from_image(img)
        features_list.append({
            'index': index,
            'image_path': str(image_path).split('/')[1].split('.')[0],
            'keypoints': keypoints,
            'descriptors': descriptors
        })

    return features_list

def extract_sift_features_from_image(image):
   query_img =cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
   sift = cv2.SIFT_create()
   query_keypoints, query_descriptors= sift.detectAndCompute(query_img, None)
   return query_keypoints, query_descriptors

def match_sift_features(query_keypoints, query_descriptors, 
                        sift_features_list, threshold=0.6):
    bf = cv2.BFMatcher()
    best_match = None
    max_matches = 0
    for features in sift_features_list:
        stored_keypoints = features['keypoints']
        stored_descriptors = features['descriptors']
        matches = bf.knnMatch(query_descriptors, stored_descriptors, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_matches.append(m)
        if len(good_matches) > max_matches:
            max_matches = len(good_matches)
            best_match = features

    return best_match
