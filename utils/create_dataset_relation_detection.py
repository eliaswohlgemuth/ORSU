import csv
import json
import os
import random

N_SAMPLES = 1000

# TODO potentially add option to create multi-class or binary classification queries
query_type = "classification"

RELATIONS = ['Cleaning', 'Drilling', 'Calibrating', 'Manipulating', 'Hammering', 'Holding', 'Touching', 'Scanning', \
             'Assisting', 'Preparing', 'CloseTo', 'LyingOn', 'Cutting', 'Cementing', 'Suturing', 'Sawing']

PRINTABLE_NAMES_TO_RELATIONS = {
    "cleaning": "Cleaning",
    "drilling": "Drilling",
    "calibrating": "Calibrating",
    "manipulating": "Manipulating",
    "hammering": "Hammering",
    "holding": "Holding",
    "touching": "Touching",
    "scanning": "Scanning",
    "assisting": "Assisting",
    "preparing": "Preparing",
    "close to": "CloseTo",
    "lying on": "LyingOn",
    "cutting": "Cutting",
    "cementing": "Cementing",
    "suturing": "Suturing",
    "sawing": "Sawing"
}

RELATIONS_TO_PRINTABLE_NAMES = {v: k for k, v in PRINTABLE_NAMES_TO_RELATIONS.items()}

PRINTABLE_NAMES_TO_SEGMENTATIONS = {
    "instrument table": "instrument_table",
    "anesthesia equipment": "ae",
    "operating table": "ot",
    "MPS station": "mps_station",
    "patient": "patient",
    "drape": "drape",
    "anesthetist": "anest",
    "circulator": "circulator",
    "assistant surgeon": "assistant_surgeon",
    "head surgeon": "head_surgeon",
    "mps": "mps",
    "nurse": "nurse",
    "drill": "drill",
    "hammer": "hammer",
    "saw": "saw",
    "tracker": "tracker",
    "robot": "mako_robot",
    "monitor": "monitor",
    "C-arm": "c_arm",
    "person": "unrelated_person"
}

SEGMENTATIONS_TO_PRINTABLE_NAMES = {v: k for k, v in PRINTABLE_NAMES_TO_SEGMENTATIONS.items()}

def main():
    samples = get_random_samples()

    with open('queries_basic_relations.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'sample_id', 'possible_answers', 'query_type', 'query', 'answer', 'image_name'])

        for key, value in samples.items():
            image_name = os.path.basename(value["image_path"])
            # NOTE can fill out query_type for potential future use
            writer.writerow([key, key, '', '', value["query"], value["answer"], image_name])


def get_random_samples():
    # create list of all relation_labels jsons
    base_relation_labels = "/home/guests/elias.wohlgemuth/dataset/001_PKA/relation_labels"
    all_relation_files = []

    lookup_table = json.load(open(os.path.join(os.path.dirname(base_relation_labels), 'timestamp_to_pcd_and_frames_list.json')))

    for filename in os.listdir(base_relation_labels):
        if filename.endswith(".json"):
            all_relation_files.append(os.path.join(base_relation_labels, filename))

    random.shuffle(all_relation_files)

    samples = dict()

    # Iterate through the randomized list of images
    query_possibilities = [style1_classification, style2_classification, style3_classification] if query_type == "classification" else [style1_binary]

    index = 0
    exit_flag = False

    for json_file in all_relation_files:
        # retreive the annotation index for image matching later on
        relation_idx = os.path.basename(json_file).split(".")[0]
        with open(json_file, "r") as f:
            json_data = json.load(f)
            for relation in json_data["rel_annotations"]:
                relation = [SEGMENTATIONS_TO_PRINTABLE_NAMES[relation[0]] if relation[0] in SEGMENTATIONS_TO_PRINTABLE_NAMES else relation[0], \
                            RELATIONS_TO_PRINTABLE_NAMES[relation[1]], \
                            SEGMENTATIONS_TO_PRINTABLE_NAMES[relation[2]] if relation[2] in SEGMENTATIONS_TO_PRINTABLE_NAMES else relation[2]]
                # randomly choose one of the following query possibilties for variety
                query, answer = random.choice(query_possibilities)(relation)

                azure_index = lookup_table[int(relation_idx)][1]['azure']
                azure_index = str(azure_index).zfill(6)

                # only using camera01 images for now
                image_folder = os.path.join(os.path.dirname(base_relation_labels), "colorimage")
                image_path = os.path.join(image_folder, f"camera01_colorimage-{azure_index}.jpg")

                samples[index] = {"image_path": image_path, "query": query, "answer": answer}
                index += 1
                if index >= N_SAMPLES:
                    exit_flag = True
                    break
        if exit_flag:
            break
    
    return samples

def style1_classification(relation):
    query = f'What is the relation between the {relation[0]} and the {relation[2]}?'
    answer = f"The {relation[0]} is {relation[1]} the {relation[2]}"
    return query, answer

def style2_classification(relation):
    query = f'What is the {relation[0]} doing with the {relation[2]}?'
    answer = f"The {relation[0]} is {relation[1]} the {relation[2]}"
    return query, answer

def style3_classification(relation):
    query = f'Is the {relation[0]} doing something with the {relation[2]}?'
    answer = f"The {relation[0]} is {relation[1]} the {relation[2]}"
    return query, answer

def style1_binary(relation):
    # produce 50/50 yes/no question
    if random.randint(0, 1) == 0:
        sampled_relation = random.choice([r for r in RELATIONS if r != relation[1]])
    else:
        sampled_relation = relation[1]
    query = f'Is the {relation[0]} {RELATIONS_TO_PRINTABLE_NAMES[sampled_relation]} the {relation[2]}?'
    answer = "yes" if relation[1] == sampled_relation else "no"
    return query, answer


if __name__ == "__main__":
    main()
