import csv
import datetime
from pathlib import Path

import decord
from decord import cpu, gpu
import numpy as np
from PIL import Image
import pandas as pd
import re
from tqdm import tqdm
from torch.utils.data import Dataset

from local_datasets import accuracy as general_accuracy
from vision_processes import forward, clear_all_consumers, instantiate_consumer


class ORmultimodalDataset(Dataset):
    def __init__(self, split, data_path="", input_type='image', image_transforms=None, fps=30, max_num_frames=30,
                 max_samples=None, start_sample=0, source_csv="queries.csv", llm_acc_metric=False, acc_prompt=None, 
                 model=None, model_name=None, max_tokens=None, **kwargs):
        """
        Args:
            split (str): Data split.
            data_path (str): Path to the data folder
            input_type (str): Type of input. One of ["image", "video"]
            image_transforms (callable, optional): Optional transform to be applied on an image. Only used if input_type
                is "image".
            fps (int): Frames per second. Only used if input_type is "video".
            max_num_frames (int): Maximum number of frames to use. Only used if input_type is "video".
            max_samples (int, optional): Maximum number of samples to load. If None, load all samples.
            start_sample (int, optional): Index of the first sample to load. If None, start from the beginning.
        """

        self.split = split
        self.data_path = Path(data_path)
        self.input_type = input_type
        self.image_transforms = image_transforms
        self.fps = fps
        self.max_num_frames = max_num_frames
        self.source_csv = source_csv
        self.llm_acc_metric = llm_acc_metric
        self.acc_prompt = acc_prompt
        self.model = model
        self.model_name = model_name
        self.max_tokens = max_tokens

        # Load questions, answers, and image ids
        with open(self.data_path / self.split / self.source_csv, 'r') as f:
            # The csv has the rows [query, answer, image_name or video_name]
            self.df = pd.read_csv(f, index_col=None, keep_default_na=False)

        if max_samples is not None:
            self.df = self.df.iloc[start_sample:start_sample + max_samples]

        self.n_samples = len(self.df)

    def get_sample_path(self, index):
        sample_name = self.df.iloc[index][f"{self.input_type}_name"]
        sample_path = self.data_path / f"{self.input_type}s" / "colorimage" / sample_name
        return sample_path

    def get_image(self, image_path):
        with open(image_path, "rb") as f:
            pil_image = Image.open(f).convert("RGB")
        if self.image_transforms:
            image = self.image_transforms(pil_image)[:3]
        else:
            image = pil_image
        return image

    def __getitem__(self, index):

        out_dict = self.df.iloc[index].to_dict()

        sample_path = self.get_sample_path(index)

        # Load and transform image
        image = self.get_image(sample_path) if self.input_type == "image" else None

        out_dict["image"] = image
        out_dict["index"] = index

        if 'extra_context' not in out_dict:
            out_dict['extra_context'] = ''

        return out_dict

    def __len__(self):
        return self.n_samples

    def accuracy(self, prediction, ground_truth, all_queries, *args):
        if not self.llm_acc_metric:
            return general_accuracy(prediction, ground_truth, *args)
        else:
            return self.accuracy_with_llm(prediction, ground_truth, all_queries)

    def accuracy_with_llm(self, prediction, ground_truth, all_queries):
        #NOTE this prompt template might need extension to more divers set of queries
        with open(self.acc_prompt) as f:
            base_prompt = f.read().strip()

        clear_all_consumers()
        # this will instantiate the Exllamav2 class, where the parameters model_id and max_new_tokens can be set in init
        instantiate_consumer(self.model, model_id=self.model_name, max_new_tokens=self.max_tokens)
        
        dummy_prompt = "i was here"
        score = 0
        #create csv for storing grading blocks later
        gradings = []
        for pred, gt, query in tqdm(zip(prediction, ground_truth, all_queries), 
                                    total=len(prediction), leave=True, desc="Calculating accuracy with LLM"):
            prompt = base_prompt.replace("INSERT QUERY", query).\
                                replace("INSERT ANSWER", str(pred)).\
                                replace("INSERT GROUND TRUTH", str(gt))
            output = forward(self.model, dummy_prompt, base_prompt=prompt)
            # only count "Score: 1" as score, otherwise 0
            if "Score: 1" in output:
                score += 1
            
            # store grading output
            # append output to prompt again
            output = output.replace("Reasoning:", "")
            extended_prompt = prompt + output
            # split at double linebreak and take the last part
            grading_block = extended_prompt.split('\n\n')[-1]
            gradings.append(grading_block)

        # write grading blocks to csv
        task = self.source_csv.split("/")[1].split(".")[0]
        now = datetime.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        csv_file = f"{task}_gradings{timestamp}.csv"
        with open(csv_file, mode='w', newline="") as file:
            writer = csv.writer(file)
            writer.writerow(["index", "grading"])
            for index, grading in enumerate(gradings):
                writer.writerow([index, grading])

        return score / len(prediction)
