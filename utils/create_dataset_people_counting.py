import csv
import os
import random
from PIL import Image
from tqdm import tqdm

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
    'unrelated_person': {'color': (255, 255, 255), 'label': 26}
}

PEOPLE = ['assistant_surgeon', 'head_surgeon', 'nurse', 'anest', 'circulator', 'mps', 'unrelated_person', 'patient']


N_SAMPLES = 1000

def main():
    # write all filenames from given folder into list
    folder01 = "/home/guests/elias.wohlgemuth/dataset/001_PKA/segmentation_export_1"
    folder04 = "/home/guests/elias.wohlgemuth/dataset/001_PKA/segmentation_export_4"
    folder05 = "/home/guests/elias.wohlgemuth/dataset/001_PKA/segmentation_export_5"
    folders = [folder01, folder04, folder05]

    samples = get_random_samples(folders)

    with open('queries_people_counting.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'sample_id', 'possible_answers', 'query_type', 'query', 'answer', 'image_name'])
        for key, value in samples.items():

            file_name = os.path.basename(value["file_path"])
            # remove "_interpolated" from file name
            file_name = file_name.replace("_interpolated", "")
            # images from colorimage folder are in jpg format
            file_name = file_name.replace("png", "jpg")
            writer.writerow([key, key, '', '', f'How many people are there?', value["count"], file_name])


def get_random_samples(folder_paths):
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

    samples = dict()

    # Iterate through the randomized list of images
    index = 0
    for image_path in tqdm(all_images, total=len(all_images)):
        if index >= N_SAMPLES:
            break

        # Open the image
        with Image.open(image_path) as img:
            count = check_matching(img)
            if count == -1:
                continue
            samples[index] = {"file_path": image_path, "count": count}
            index += 1
    
    return samples


def check_matching(img):
    # outputs if label pixel value is present in image
    # create a set of unique values in img
    unique_values = set(img.getdata())
    if TRACK_TO_METAINFO['unrelated_person']['label'] in unique_values:
        # if unrelated person is present, check SG for multiple?
        # for now skip images with unrelated person, as they are not essential for counting query
        return -1
    if TRACK_TO_METAINFO['patient']['label'] in unique_values:
        # for now skip images with patient, since often just the knee is visible -> advanced 
        return -1
    matches = []
    [matches.append(TRACK_TO_METAINFO[role]["label"] in unique_values) for role in PEOPLE]
    # sum over matches, where True is 1 and False is 0
    return sum(matches)


if __name__ == "__main__":
    main()