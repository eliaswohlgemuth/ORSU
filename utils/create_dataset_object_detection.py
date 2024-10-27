import csv
import os
import random
from PIL import Image

TRACK_TO_METAINFO = {
    'instrument_table': {'color': (255, 51, 153), 'label': 1},
    'ae': {'color': (0, 0, 255), 'label': 2},
    'ot': {'color': (255, 255, 0), 'label': 3},
    'mps_station': {'color': (133, 0, 133), 'label': 4},
    'patient': {'color': (255, 0, 0), 'label': 5},
    'drape': {'color': (183, 91, 255), 'label': 6},
    'anest': {'color': (177, 255, 110), 'label': 7},
    'circulator': {'color': (255, 128, 0), 'label': 8},
    'assistant_surgeon': {'color': (116, 166, 116), 'label': 9},
    'head_surgeon': {'color': (76, 161, 245), 'label': 10},
    'mps': {'color': (125, 100, 25), 'label': 11},
    'nurse': {'color': (128, 255, 0), 'label': 12},
    'drill': {'color': (0, 255, 128), 'label': 13},  # Changed
    'hammer': {'color': (204, 0, 0), 'label': 15},
    'saw': {'color': (0, 255, 234), 'label': 16},
    'tracker': {'color': (255, 128, 128), 'label': 17},  # Changed
    'mako_robot': {'color': (60, 75, 255), 'label': 18},  # Changed
    'monitor': {'color': (255, 255, 128), 'label': 24},  # Changed
    'c_arm': {'color': (0, 204, 128), 'label': 25},  # Changed
    # 'unrelated_person': {'color': (255, 255, 255), 'label': 26}
}

PRINTABLE_NAMES = {
    "instrument_table": "instrument table",
    "ae": "anesthesia equipment",
    "ot": "operating table",
    "mps_station": "MPS station",
    "patient": "patient",
    "drape": "drape",
    "anest": "anesthetist",
    "circulator": "circulator",
    "assistant_surgeon": "assistant surgeon",
    "head_surgeon": "head surgeon",
    "mps": "mps",
    "nurse": "nurse",
    "drill": "drill",
    "hammer": "hammer",
    "saw": "saw",
    "tracker": "tracker",
    "mako_robot": "robot",
    "monitor": "monitor",
    "c_arm": "C-arm",
    "unrelated_person": "unrelated person"
}

N_SAMPLES = 20

def main():
    # write all filenames from given folder into list
    folder01 = "/home/guests/elias.wohlgemuth/dataset/001_PKA/segmentation_export_1"
    folder04 = "/home/guests/elias.wohlgemuth/dataset/001_PKA/segmentation_export_4"
    folder05 = "/home/guests/elias.wohlgemuth/dataset/001_PKA/segmentation_export_5"
    folders = [folder01, folder04, folder05]

    detections_true = dict()
    for key in TRACK_TO_METAINFO.keys():
        detections_true[key] = []

    detections_false = dict()
    for key in TRACK_TO_METAINFO.keys():
        detections_false[key] = []

    # create the samples
    get_random_samples(detections_true, detections_false, folders, N_SAMPLES)

    # create csv
    with open('obj_detect.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'sample_id', 'possible_answers', 'query_type', 'query', 'answer', 'image_name'])
        index = 0

        # true detections
        for key, file_paths in detections_true.items():
            for file_path in file_paths:
                file_name = os.path.basename(file_path)
                # remove "_interpolated" from file name
                file_name = file_name.replace("_interpolated", "")
                # images from colorimage folder are in jpg format
                file_name = file_name.replace("png", "jpg")
                writer.writerow([index, index, '', '', f'Is there a {PRINTABLE_NAMES[key]}?', 'yes', file_name])
                index += 1
        # false detections
        for key, file_paths in detections_false.items():
            # add negative detections to csv
            for file_path in file_paths:
                file_name = os.path.basename(file_path)
                # remove "_interpolated" from file name
                file_name = file_name.replace("_interpolated", "")
                # images from colorimage folder are in jpg format
                file_name = file_name.replace("png", "jpg")
                writer.writerow([index, index, '', '', f'Is there a {PRINTABLE_NAMES[key]}?', 'no', file_name])
                index += 1


def get_random_samples(detections_positive, detections_negative, folder_paths, n_samples):
    # List to store all image paths from the folders
    all_images = []

    # Gather all image file paths from the specified folders
    for folder in folder_paths:
        for filename in os.listdir(folder):
            if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                full_path = os.path.join(folder, filename)
                all_images.append(full_path)

    # Randomize the list of images
    random.shuffle(all_images)

    # Iterate through the randomized list of images
    for i, image_path in enumerate(all_images):
        if i % 500 == 0:
            print(f"Processed {i} images")

        # check if every value in detections has length of n_samples
        if all([len(value) == n_samples for value in detections_positive.values()]) \
            and all([len(value) == n_samples for value in detections_negative.values()]):
            break   #done

        # Open the image
        with Image.open(image_path) as img:
            # Check the specified condition
            condition_mask_positive, condition_mask_negative = check_matching(img)
            for key in TRACK_TO_METAINFO.keys():
                if condition_mask_positive[key] and len(detections_positive[key]) < n_samples:
                    detections_positive[key].append(image_path)  
                if condition_mask_negative[key] and len(detections_negative[key]) < n_samples:
                    detections_negative[key].append(image_path)  

def check_matching(img):
    # outputs if label pixel value is present in image
    # NOTE alternatively could check if there is a number of pixels with the label value, to get some detection threshold
    unique_values = set(img.getdata())
    mask_positive = {key: value["label"] in unique_values for key, value in TRACK_TO_METAINFO.items()}
    mask_negative = {key: value["label"] not in unique_values for key, value in TRACK_TO_METAINFO.items()}
    return mask_positive, mask_negative


if __name__ == "__main__":
    main()