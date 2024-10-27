import csv
import json

# head row: index,sample_id,possible_answers,query_type,query,answer,image_name
# open json file
frames_lookup = json.load(open('/home/guests/elias.wohlgemuth/dataset/001_PKA/timestamp_to_pcd_and_frames_list.json'))

azure_frames = []
for i, _ in enumerate(frames_lookup):
    azure_frames.append(frames_lookup[i][1]['azure'])

with open('data/OR-multimodal/queries_generated/queries_operation_phase.csv', "w") as f:
    # write head row
    writer = csv.writer(f)
    writer.writerow(['index', 'sample_id', 'possible_answers', 'query_type', 'query', 'answer', 'image_name'])
    for i in range(len(azure_frames)):
        writer.writerow([i, i, '', '', 'What is the current operation phase?', '', f"camera01_colorimage-{azure_frames[i]}.jpg"])