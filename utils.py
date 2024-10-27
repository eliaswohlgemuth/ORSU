from collections import Counter
from datetime import datetime
import json
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import pathlib
import random
import sys
import shutil
import time
import torch
from PIL import Image
from torchvision import transforms
from torchvision.utils import draw_bounding_boxes as tv_draw_bounding_boxes
from torchvision.utils import make_grid
from typing import Union

clip_stats = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)

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

PRINTABLE_NAMES_TO_SEGMENTATIONS = {
    "instrument table": "instrument_table",
    "anesthesia equipment": "ae",
    "operating table": "ot",
    "mps station": "mps_station",
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
    "c-arm": "c_arm",
    "person": "person", # not a segmentation label but needed for ground truth for people counting
}

SEGMENTATIONS_TO_PRINTABLE_NAMES = {v: k for k, v in PRINTABLE_NAMES_TO_SEGMENTATIONS.items()}

PRINTABLE_NAMES_TO_RELATION_OBJECTS = {
    **PRINTABLE_NAMES_TO_SEGMENTATIONS,
    "instrument": "instrument"
}

RELATION_OBJECTS_TO_PRINTABLE_NAMES = {v: k for k, v in PRINTABLE_NAMES_TO_RELATION_OBJECTS.items()}

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


def is_interactive() -> bool:
    try:
        from IPython import get_ipython
        if get_ipython() is not None:
            return True
        else:
            return False
    except NameError:
        return False  # Probably standard Python interpreter


def denormalize(images, means=(0.485, 0.456, 0.406), stds=(0.229, 0.224, 0.225)):
    means = torch.tensor(means).reshape(1, 3, 1, 1)
    stds = torch.tensor(stds).reshape(1, 3, 1, 1)
    return images * stds + means


def show_batch(batch, stats=clip_stats):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.set_xticks([])
    ax.set_yticks([])
    denorm_images = denormalize(batch, *stats)
    ax.imshow(make_grid(denorm_images[:64], nrow=8).permute(1, 2, 0).clamp(0, 1))


def show_batch_from_dl(dl):
    for images, labels in dl:
        show_batch(images)
        print(labels[:64])
        break


def show_single_image(image, denormalize_stats=None, bgr_image=False, save_path=None, size='small', bbox_info=None):
    if not is_interactive():
        import matplotlib
        matplotlib.use("module://imgcat")
    if size == 'size_img':
        figsize = (image.shape[2] / 100, image.shape[1] / 100)  # The default dpi of plt.savefig is 100
    elif size == 'small':
        figsize = (4, 4)
    else:
        figsize = (12, 12)

    fig = plt.figure(figsize=figsize)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xticks([])
    ax.set_yticks([])

    if bbox_info is not None:
        image = draw_bounding_boxes(image, bbox_info['bboxes'], labels=bbox_info['labels'], colors=bbox_info['colors'],
                                    width=5)

    if isinstance(image, torch.Tensor):
        image = image.detach().cpu()
        if denormalize_stats is not None:
            image = denormalize(image.unsqueeze(0), *denormalize_stats)
        if image.dtype == torch.float32:
            image = image.clamp(0, 1)
        ax.imshow(image.squeeze(0).permute(1, 2, 0))
    else:
        if bgr_image:
            image = image[..., ::-1]
        ax.imshow(image)

    if save_path is None:
        plt.show()
    # save image if save_path is provided
    if save_path is not None:
        # make path if it does not exist
        if not os.path.exists(os.path.dirname(save_path)):
            os.makedirs(os.path.dirname(save_path))
        plt.savefig(save_path)


def draw_bounding_boxes(
        image: Union[torch.Tensor, Image.Image],
        bboxes: Union[list, torch.Tensor],
        width: int = 5,
        **kwargs
):
    """
    Wrapper around torchvision.utils.draw_bounding_boxes
    bboxes: [xmin, ymin, xmax, ymax]
    :return:
    """
    if isinstance(image, Image.Image):
        if type(image) == Image.Image:
            image = transforms.ToTensor()(image)
    if isinstance(bboxes, list):
        bboxes = torch.tensor(bboxes)

    image = (image * 255).to(torch.uint8).cpu()
    height = image.shape[1]
    bboxes = torch.stack([bboxes[:, 0], height - bboxes[:, 3], bboxes[:, 2], height - bboxes[:, 1]], dim=1)
    return tv_draw_bounding_boxes(image, bboxes, width=width, **kwargs)


def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_index_from_sample_id(sample_id, dataset):
    df = dataset.df
    return np.arange(df.shape[0])[df.index == sample_id]


def save_json(data: dict, path: Union[str, pathlib.Path]):
    if isinstance(path, str):
        path = pathlib.Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True)
    if path.suffix != '.json':
        path = path.with_suffix('.json')
    with open(path, 'w') as f:
        json.dump(data, f, indent=4, sort_keys=True)


def load_json(path: Union[str, pathlib.Path]):
    if isinstance(path, str):
        path = pathlib.Path(path)
    if path.suffix != '.json':
        path = path.with_suffix('.json')
    with open(path, 'r') as f:
        data = json.load(f)
    return data


def make_print_safe(string: str) -> str:
    return string.replace(r'[', r'\[')


def sprint(string: str):
    print(make_print_safe(string))


def print_full_df(df):
    with pd.option_context('display.max_rows', None, 'display.max_columns', None):  # more options can be specified also
        if is_interactive():
            display(df)
        else:
            print(df)


def code_to_paste(code):
    print('\n'.join([c[4:] for c in code.split('\n')[1:]]).replace('image', 'ip').replace('return ', ''))


def log_files(config_file):
    base_dir = os.path.dirname(os.path.abspath(__file__))

    config_name = config_file.dataset.data_path.split('/')[-1].lower()

    if "gqa" in config_name:
        files_to_copy = ['main_batch.py', 'vision_models.py', f'configs/benchmarks/{config_name}.yaml']
    else:
        files_to_copy = ['main_batch.py', 'vision_models.py', f'configs/{config_name}.yaml',
                         "image_patch.py", config_file.codex.prompt]
    files_to_copy = [os.path.join(base_dir, file) for file in files_to_copy]

    destination_dir = 'outputs'

    current_date = datetime.now().strftime('%Y-%m-%d')
    current_time = datetime.now().strftime('%H-%M-%S')

    date_dir = os.path.join(destination_dir, current_date)
    time_dir = os.path.join(date_dir, current_time)

    os.makedirs(time_dir, exist_ok=True)

    for file_path in files_to_copy:
        if os.path.isfile(file_path):
            shutil.copy(file_path, time_dir)
        else:
            print(f"File not found: {file_path}")


class HiddenPrints:
    hide_prints = False

    def __init__(self, model_name=None, console=None, use_newline=True):
        self.model_name = model_name
        self.console = console
        self.use_newline = use_newline
        self.tqdm_aux = None

    def __enter__(self):
        if self.hide_prints:
            import tqdm  # We need to do an extra step to hide tqdm outputs. Does not work in Jupyter Notebooks.

            def nop(it, *a, **k):
                return it

            self.tqdm_aux = tqdm.tqdm
            tqdm.tqdm = nop

            if self.model_name is not None:
                self.console.print(f'Loading {self.model_name}...')
            self._original_stdout = sys.stdout
            self._original_stderr = sys.stderr
            sys.stdout = open(os.devnull, 'w')
            # May not be what we always want, but some annoying warnings end up to stderr
            sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.hide_prints:
            sys.stdout.close()
            sys.stdout = self._original_stdout
            sys.stdout = self._original_stderr
            if self.model_name is not None:
                self.console.print(f'{self.model_name} loaded ')
            import tqdm
            tqdm.tqdm = self.tqdm_aux
