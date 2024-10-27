from __future__ import annotations

import numpy as np
import re
import torch
from dateutil import parser as dateparser
import os
from PIL import Image
from pydub import AudioSegment
from rich.console import Console
from torchvision import transforms
from torchvision.ops import box_iou
from typing import Union, List, Tuple, Optional
from word2number import w2n

from utils import show_single_image, load_json, TRACK_TO_METAINFO, PRINTABLE_NAMES_TO_SEGMENTATIONS, SEGMENTATIONS_TO_PRINTABLE_NAMES, \
    PRINTABLE_NAMES_TO_RELATION_OBJECTS, RELATION_OBJECTS_TO_PRINTABLE_NAMES, PRINTABLE_NAMES_TO_RELATIONS, RELATIONS_TO_PRINTABLE_NAMES
from vision_processes import forward, config
import vision_processes

FRAMES_LOOKUP_TABLE = load_json(os.path.join(config.dataset.data_path, 'timestamp_to_pcd_and_frames_list.json'))

console = Console(highlight=False)


class ImagePatch:
    """A Python class containing a crop of an image centered around a particular object, as well as relevant
    information.
    Attributes
    ----------
    cropped_image : array_like
        An array-like of the cropped image taken from the original image.
    left : int
        An int describing the position of the left border of the crop's bounding box in the original image.
    lower : int
        An int describing the position of the bottom border of the crop's bounding box in the original image.
    right : int
        An int describing the position of the right border of the crop's bounding box in the original image.
    upper : int
        An int describing the position of the top border of the crop's bounding box in the original image.

    Methods
    -------
    find(object_name: str)->List[ImagePatch]
        Returns a list of new ImagePatch objects containing crops of the image centered around any objects found in the
        image matching the object_name.
    exists(object_name: str)->bool
        Returns True if the object specified by object_name is found in the image, and False otherwise.
    verify_property(property: str)->bool
        Returns True if the property is met, and False otherwise.
    best_text_match(option_list: List[str], prefix: str)->str
        Returns the string that best matches the image.
    simple_query(question: str=None)->str
        Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?".
    compute_depth()->float
        Returns the median depth of the image crop.
    crop(left: int, lower: int, right: int, upper: int)->ImagePatch
        Returns a new ImagePatch object containing a crop of the image at the given coordinates.
    """

    audio_anchors = None
    audio_classes = ["sawing", "drilling", "hammering"]

    def __init__(self, image: Union[Tuple[Union[Image.Image, torch.Tensor, np.ndarray], str], Union[Image.Image, torch.Tensor, np.ndarray]], left: int = None, lower: int = None,
                 right: int = None, upper: int = None, parent_left=0, parent_lower=0, queues=None,
                 parent_img_patch=None):
        """Initializes an ImagePatch object by cropping the image at the given coordinates and stores the coordinates as
        attributes. If no coordinates are provided, the image is left unmodified, and the coordinates are set to the
        dimensions of the image.

        Parameters
        -------
        image : Tuple[array_like, str] or array_like
            An array-like of the image, or a tuple containing the image and the image's name. Latter is required when using the ground truth.
        left : int
            An int describing the position of the left border of the crop's bounding box in the original image.
        lower : int
            An int describing the position of the bottom border of the crop's bounding box in the original image.
        right : int
            An int describing the position of the right border of the crop's bounding box in the original image.
        upper : int
            An int describing the position of the top border of the crop's bounding box in the original image.

        """
        if isinstance(image, tuple):
            image, self.image_name = image

        if isinstance(image, Image.Image):
            image = transforms.ToTensor()(image)
        elif isinstance(image, np.ndarray):
            image = torch.tensor(image).permute(1, 2, 0)
        elif isinstance(image, torch.Tensor) and image.dtype == torch.uint8:
            image = image / 255

        if left is None and right is None and upper is None and lower is None:
            self.cropped_image = image
            self.left = 0
            self.lower = 0
            self.right = image.shape[2]  # width
            self.upper = image.shape[1]  # height
        else:
            self.cropped_image = image[:, image.shape[1]-upper:image.shape[1]-lower, left:right]
            self.left = left + parent_left
            self.upper = upper + parent_lower
            self.right = right + parent_left
            self.lower = lower + parent_lower

        self.height = self.cropped_image.shape[1]
        self.width = self.cropped_image.shape[2]

        self.cache = {}
        self.queues = (None, None) if queues is None else queues

        self.parent_img_patch = parent_img_patch

        self.horizontal_center = (self.left + self.right) / 2
        self.vertical_center = (self.lower + self.upper) / 2

        if self.cropped_image.shape[1] == 0 or self.cropped_image.shape[2] == 0:
            raise Exception("ImagePatch has no area")

        self.possible_options = load_json('./useful_lists/possible_options.json')

        if ImagePatch.audio_anchors is None:
            ImagePatch.audio_anchors = ImagePatch._prepare_anchors(ImagePatch.audio_classes)

    def forward(self, model_name, *args, **kwargs):
        return forward(model_name, *args, queues=self.queues, **kwargs)

    @property
    def original_image(self):
        if self.parent_img_patch is None:
            return self.cropped_image
        else:
            return self.parent_img_patch.original_image

    def find(self, object_name: str) -> list[ImagePatch]:
        """Returns a list of ImagePatch objects matching object_name contained in the crop if any are found.
        Otherwise, returns an empty list.
        Parameters
        ----------
        object_name : str
            the name of the object to be found

        Returns
        -------
        List[ImagePatch]
            a list of ImagePatch objects matching object_name contained in the crop
        """
        object_name = object_name.lower()
        # implement option to use ground truth from segmentation
        if config.ground_truth.find and object_name in PRINTABLE_NAMES_TO_SEGMENTATIONS.keys():
            all_object_coordinates = object_detection_from_segmentation(self.image_name, object_name)

        elif object_name in ["object", "objects"]:
            all_object_coordinates = self.forward('maskrcnn', self.cropped_image)[0]
        else:

            if object_name == 'person':
                object_name = 'people'  # GLIP does better at people than person

            all_object_coordinates = self.forward('glip', self.cropped_image, object_name)
        if len(all_object_coordinates) == 0:
            return []

        threshold = config.ratio_box_area_to_image_area
        if threshold > 0:
            area_im = self.width * self.height
            all_areas = torch.tensor([(coord[2]-coord[0]) * (coord[3]-coord[1]) / area_im
                                      for coord in all_object_coordinates])
            mask = all_areas > threshold
            # if not mask.any():
            #     mask = all_areas == all_areas.max()  # At least return one element
            all_object_coordinates = all_object_coordinates[mask]


        return [self.crop(*coordinates) for coordinates in all_object_coordinates]

    def exists(self, object_name) -> bool:
        """Returns True if the object specified by object_name is found in the image, and False otherwise.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        """
        if object_name.isdigit() or object_name.lower().startswith("number"):
            object_name = object_name.lower().replace("number", "").strip()

            object_name = w2n.word_to_num(object_name)
            answer = self.simple_query("What number is written in the image (in digits)?")
            return w2n.word_to_num(answer) == object_name

        patches = self.find(object_name)

        if not config.ground_truth.find:
            filtered_patches = []
            for patch in patches:
                if "yes" in patch.simple_query(f"Is this a {object_name}?"):
                    filtered_patches.append(patch)
        else:
            filtered_patches = patches
        return len(filtered_patches) > 0

    def _score(self, category: str, negative_categories=None, model='clip') -> float:
        """
        Returns a binary score for the similarity between the image and the category.
        The negative categories are used to compare to (score is relative to the scores of the negative categories).
        """
        if model == 'clip':
            res = self.forward('clip', self.cropped_image, category, task='score',
                               negative_categories=negative_categories)
        elif model == 'tcl':
            res = self.forward('tcl', self.cropped_image, category, task='score')
        else:  # xvlm
            task = 'binary_score' if negative_categories is not None else 'score'
            res = self.forward('xvlm', self.cropped_image, category, task=task, negative_categories=negative_categories)
            res = res.item()

        return res

    def _detect(self, category: str, thresh, negative_categories=None, model='clip') -> bool:
        return self._score(category, negative_categories, model) > thresh

    def verify_property(self, object_name: str, attribute: str) -> bool:
        """Returns True if the object possesses the property, and False otherwise.
        Differs from 'exists' in that it presupposes the existence of the object specified by object_name, instead
        checking whether the object possesses the property.
        Parameters
        -------
        object_name : str
            A string describing the name of the object to be found in the image.
        attribute : str
            A string describing the property to be checked.
        """
        name = f"{attribute} {object_name}"
        model = config.verify_property.model
        negative_categories = [f"{att} {object_name}" for att in self.possible_options['attributes']]
        if model == 'clip':
            return self._detect(name, negative_categories=negative_categories,
                                thresh=config.verify_property.thresh_clip, model='clip')
        elif model == 'tcl':
            return self._detect(name, thresh=config.verify_property.thresh_tcl, model='tcl')
        else:  # 'xvlm'
            return self._detect(name, negative_categories=negative_categories,
                                thresh=config.verify_property.thresh_xvlm, model='xvlm')

    def best_text_match(self, option_list: list[str] = None, prefix: str = None) -> str:
        """Returns the string that best matches the image.
        Parameters
        -------
        option_list : str
            A list with the names of the different options
        prefix : str
            A string with the prefixes to append to the options
        """
        option_list_to_use = option_list
        if prefix is not None:
            option_list_to_use = [prefix + " " + option for option in option_list]

        model_name = config.best_match_model
        image = self.cropped_image
        text = option_list_to_use
        if model_name in ('clip', 'tcl'):
            selected = self.forward(model_name, image, text, task='classify')
        elif model_name == 'xvlm':
            res = self.forward(model_name, image, text, task='score')
            res = res.argmax().item()
            selected = res
        else:
            raise NotImplementedError

        return option_list[selected]

    def simple_query(self, question: str) -> str:
        """Returns the answer to a basic question asked about the image. If no question is provided, returns the answer
        to "What is this?". The questions are about basic perception, and are not meant to be used for complex reasoning
        or external knowledge.
        Parameters
        -------
        question : str
            A string describing the question to be asked.
        """
        if config.ground_truth.relation:
            # extract potential segmentation labels from questions
            question = question.lower()
            contained_labels = [label for label in PRINTABLE_NAMES_TO_RELATION_OBJECTS.keys() if label in question]

            # NOTE currently, if query contains instrument table, this gets matched with instrument table and instrument
            # therefore there will be three labels matches, which is not intended. As instrument is not inteded for instrument table it gets cut off
            # this is a workaround for now
            contained_labels = contained_labels[:2]

            # assign the label that comes first in question to object 1
            label_positions = {label: question.find(label) for label in contained_labels}
            if len(label_positions) != 0:
                object_1 = min(label_positions, key=label_positions.get)
                object_2 = max(label_positions, key=label_positions.get)

        # NOTE this can falsely trigger if the question is complex and contains two seg labels among multiple objects, but is not about a single relation
        # this looks ugly but should help to avoid false triggers
        # if question is not about relation, use vision model, as GT does not apply then
        if config.ground_truth.relation and len(contained_labels) == 2:
            return relation_from_ground_truth_SG(object_1, object_2, question, self.image_name)
        else:
            available_models = list(vision_processes.consumers.keys())
            if 'llava' in available_models:
                return self.forward('llava', question, self.cropped_image)
            elif 'blip' in available_models:
                return self.forward('blip', self.cropped_image, question, task='qa')
            else:
                raise ValueError("No suitable model available to process simple_query().")
        
    def get_operation_phase(self) -> str:
        """"
        Returns the operation phase at the current time.
        Looks up the timestamp given the index in image_name, assuming the image is from the azure camera.
        """
        # convert the index from the image to the index of simstation
        azure_idx = int(self.image_name.split("-")[1].split(".")[0])
        # TODO from azure to simstation is always a constant offset
        azure_to_simstation_offset = int(FRAMES_LOOKUP_TABLE[0][1]["simstation"]) - int(FRAMES_LOOKUP_TABLE[0][1]["azure"])
        simstation_idx_int = azure_idx + azure_to_simstation_offset
        simstation_idx = str(simstation_idx_int).zfill(6)
        
        phases = {
            # position: left, lower, right, upper
            0: {"name": "Fallplanung", "position": [140, 686, 274, 712]},
            1: {"name": "RIO-Kontrolle vor OP", "position": [274, 686, 407, 712]},
            2: {"name": "Knochenregistrierung", "position": [407, 686, 540, 712]},
            3: {"name": "Intra-OP Planung", "position": [540, 686, 674, 712]},
            4: {"name": "Intra-OP Knochenvorbereitung", "position": [674, 686, 808, 712]},
            5: {"name": "Abschluss des Falls", "position": [808, 686, 940, 712]},
        }
        
        # load simstation image
        img = Image.open(f"{config.dataset.data_path}/images/simstation/camera01_{simstation_idx}.jpg")

        detections = {}

        for key, value in phases.items():
            image_patch = ImagePatch((img, f"camera01_{simstation_idx}.jpg"), *value["position"])
            detections[key] = image_patch.best_text_match(["gray", "green", "blue"])

        match_idx = [key for key, value in detections.items() if value == "green"]

        if len(match_idx) >= 2:
            # add second test with verify_property as some detections are false positives
            re_detections = {}
            for idx in match_idx:
                image_patch = ImagePatch((img, f"camera01_{simstation_idx}.jpg"), *phases[idx]["position"])
                re_detections[idx] = image_patch.verify_property("area", "green")

            match_idx = [key for key, value in re_detections.items() if value]

        # return the operation phase matching the index
        if len(match_idx) == 0:
            return ""
        elif len(match_idx) == 1:
            return phases[match_idx[0]]["name"]
        elif len(match_idx) >= 2:
            console.print(f"Multiple operation phases: detected {len(match_idx)} phases at simstation index {simstation_idx}. Returning earliest phase.")
            return phases[match_idx[1]]["name"]
    
    def analyse_audio(self, query_sound: str) -> bool:
        """Return if a sound is present at the time of the image.
        Looks up the timestamp given the index in image_name, assuming the image is from the azure camera.
        A window of 10 seconds is used to check for the sound.
        Parameters
        -------
        query_sound : str
            One of the following:
                -sawing
                -drilling
                -hammering
        """
        # NOTE CLAP model cannot distinguish between sawing and drilling, i.e. predicts drilling for both
        # therefore few-shot-classification by calulating similarity between query and template embeddings is performed

        if query_sound not in self.audio_classes:
            raise ValueError(f"Query sound {query_sound} not in classes {self.audio_classes}")
        
        class_anchors = ImagePatch.audio_anchors
        
        # NOTE assuming the image is from the azure camera
        azure_idx = int(self.image_name.split("-")[1].split(".")[0])
        azure_offset = int(FRAMES_LOOKUP_TABLE[0][1]["azure"])
        if azure_idx < azure_offset:
            raise ValueError("No audio data available at this timepoint.")

        # NOTE assuming using audio from audio_takes/
        # determine corresponding timestamp in audio data
        timestamp = azure_idx - azure_offset    # timestamp in seconds
        start_time = timestamp - 5
        end_time = timestamp + 5

        # Load audio data
        # NOTE use this in standard case
        # currently all audio benchmark samples are from 001_PKA.mp3
        # audio = AudioSegment.from_mp3(f"{config.dataset.data_path}/audio/001_PKA.mp3")
        # assert audio.frame_rate == 48000, "Audio snippet must be resampled to 48kHz"

        # audio_snippet_pre = audio[start_time * 1000:timestamp * 1000]
        # audio_snippet_post = audio[timestamp * 1000:end_time * 1000]

        # NOTE use this for demo script, as full audio is not published
        audio_snippet_pre = AudioSegment.from_mp3(f"{config.dataset.data_path}/audio/audio_takes/001_PKA_{start_time}.mp3")
        audio_snippet_post = AudioSegment.from_mp3(f"{config.dataset.data_path}/audio/audio_takes/001_PKA_{end_time - 5}.mp3")
        assert audio_snippet_pre.frame_rate == 48000, "Audio snippet must be resampled to 48kHz"
        assert audio_snippet_post.frame_rate == 48000, "Audio snippet must be resampled to 48kHz"

        audio_snippets = [audio_snippet_pre, audio_snippet_post]
        audio_snippets_np = [np.array(audio_snippet.get_array_of_samples()) for audio_snippet in audio_snippets]

        # Perform few-shot audio classification
        normalized_logits = self.forward('clap', audio_snippets_np, class_anchors)

        # thresholds = [0.98, 1.038, 1.059]  # thresholds for sawing, drilling, hammering obtained by optimizing f1 score on val set
        thresholds = [0.969, 1.022, 1.033]  # thresholds for sawing, drilling, hammering obtained by optimizing recall on val set
        query_sound_index = self.audio_classes.index(query_sound)

        return any([snippet_scores[query_sound_index] > thresholds[query_sound_index] for snippet_scores in normalized_logits])
    
    @classmethod
    def _prepare_anchors(cls, classes) -> dict:
        """Prepare audio anchors for few-shot audio classification.
        Creates a dictionary with class names as keys and a list of audio snippets as numpy arrays as values."""
        template_dir = f"{config.dataset.data_path}/audio/class_templates"
        n_templates = 10

        # n_templates templates are stored as class anchors (5 seconds each)
        class_anchors = {c: [] for c in classes}

        template_files = os.listdir(template_dir)
        for c in classes:
            # get all mp3 files in template_dir starting with class name
            class_files = sorted([file for file in template_files if file.startswith(c)])[:n_templates]
            for file in class_files:
                audio = AudioSegment.from_mp3(f"{template_dir}/{file}")
                assert audio.frame_rate == 48000, "Audio snippet must be resampled to 48kHz"
                class_anchors[c].append(np.array(audio.get_array_of_samples()))
        
        assert all(len(class_anchors[c]) == n_templates for c in classes), "Not all classes have the required number of templates"
        return class_anchors

    def compute_depth(self):
        """Returns the median depth of the image crop
        Parameters
        ----------
        Returns
        -------
        float
            the median depth of the image crop
        """
        original_image = self.original_image
        depth_map = self.forward('depth', original_image)
        depth_map = depth_map[original_image.shape[1]-self.upper:original_image.shape[1]-self.lower,
                              self.left:self.right]
        return depth_map.median()  # Ideally some kind of mode, but median is good enough for now

    def crop(self, left: int, lower: int, right: int, upper: int) -> ImagePatch:
        """Returns a new ImagePatch containing a crop of the original image at the given coordinates.
        Parameters
        ----------
        left : int
            the position of the left border of the crop's bounding box in the original image
        lower : int
            the position of the bottom border of the crop's bounding box in the original image
        right : int
            the position of the right border of the crop's bounding box in the original image
        upper : int
            the position of the top border of the crop's bounding box in the original image

        Returns
        -------
        ImagePatch
            a new ImagePatch containing a crop of the original image at the given coordinates
        """
        # make all inputs ints
        left = int(left)
        lower = int(lower)
        right = int(right)
        upper = int(upper)

        if config.crop_larger_margin:
            left = max(0, left - 10)
            lower = max(0, lower - 10)
            right = min(self.width, right + 10)
            upper = min(self.height, upper + 10)

        # TODO add current image_name to the new ImagePatch, as in more complex code ImagePatch is used further down the road
        return ImagePatch((self.cropped_image, self.image_name), left, lower, right, upper, self.left, self.lower, queues=self.queues,
                          parent_img_patch=self)

    def overlaps_with(self, left, lower, right, upper):
        """Returns True if a crop with the given coordinates overlaps with this one,
        else False.
        Parameters
        ----------
        left : int
            the left border of the crop to be checked
        lower : int
            the lower border of the crop to be checked
        right : int
            the right border of the crop to be checked
        upper : int
            the upper border of the crop to be checked

        Returns
        -------
        bool
            True if a crop with the given coordinates overlaps with this one, else False
        """
        return self.left <= right and self.right >= left and self.lower <= upper and self.upper >= lower

    def llm_query(self, question: str, long_answer: bool = True) -> str:
        return llm_query(question, None, long_answer)

    def print_image(self, size: tuple[int, int] = None):
        show_single_image(self.cropped_image, size)

    def __repr__(self):
        return "ImagePatch({}, {}, {}, {})".format(self.left, self.lower, self.right, self.upper)


def best_image_match(list_patches: list[ImagePatch], content: List[str], return_index: bool = False) -> \
        Union[ImagePatch, None]:
    """Returns the patch most likely to contain the content.
    Parameters
    ----------
    list_patches : List[ImagePatch]
    content : List[str]
        the object of interest
    return_index : bool
        if True, returns the index of the patch most likely to contain the object

    Returns
    -------
    int
        Patch most likely to contain the object
    """
    if len(list_patches) == 0:
        return None

    model = config.best_match_model

    scores = []
    for cont in content:
        if model == 'clip':
            res = list_patches[0].forward(model, [p.cropped_image for p in list_patches], cont, task='compare',
                                          return_scores=True)
        else:
            res = list_patches[0].forward(model, [p.cropped_image for p in list_patches], cont, task='score')
        scores.append(res)
    scores = torch.stack(scores).mean(dim=0)
    scores = scores.argmax().item()  # Argmax over all image patches

    if return_index:
        return scores
    return list_patches[scores]


def distance(patch_a: Union[ImagePatch, float], patch_b: Union[ImagePatch, float]) -> float:
    """
    Returns the distance between the edges of two ImagePatches, or between two floats.
    If the patches overlap, it returns a negative distance corresponding to the negative intersection over union.
    """

    if isinstance(patch_a, ImagePatch) and isinstance(patch_b, ImagePatch):
        a_min = np.array([patch_a.left, patch_a.lower])
        a_max = np.array([patch_a.right, patch_a.upper])
        b_min = np.array([patch_b.left, patch_b.lower])
        b_max = np.array([patch_b.right, patch_b.upper])

        u = np.maximum(0, a_min - b_max)
        v = np.maximum(0, b_min - a_max)

        dist = np.sqrt((u ** 2).sum() + (v ** 2).sum())

        if dist == 0:
            box_a = torch.tensor([patch_a.left, patch_a.lower, patch_a.right, patch_a.upper])[None]
            box_b = torch.tensor([patch_b.left, patch_b.lower, patch_b.right, patch_b.upper])[None]
            dist = - box_iou(box_a, box_b).item()

    else:
        dist = abs(patch_a - patch_b)

    return dist


def bool_to_yesno(bool_answer: bool) -> str:
    """Returns a yes/no answer to a question based on the boolean value of bool_answer.
    Parameters
    ----------
    bool_answer : bool
        a boolean value

    Returns
    -------
    str
        a yes/no answer to a question based on the boolean value of bool_answer
    """
    return "yes" if bool_answer else "no"


def llm_query(query, context=None, long_answer=True, queues=None):
    """Answers a text question using GPT-3. The input question is always a formatted string with a variable in it.

    Parameters
    ----------
    query: str
        the text question to ask. Must not contain any reference to 'the image' or 'the photo', etc.
    """
    if long_answer:
        return forward(model_name='gpt3_general', prompt=query, queues=queues)
    else:
        return forward(model_name='gpt3_qa', prompt=[query, context], queues=queues)


def process_guesses(prompt, guess1=None, guess2=None, queues=None):
    return forward(model_name='gpt3_guess', prompt=[prompt, guess1, guess2], queues=queues)


def coerce_to_numeric(string, no_string=False):
    """
    This function takes a string as input and returns a numeric value after removing any non-numeric characters.
    If the input string contains a range (e.g. "10-15"), it returns the first value in the range.
    # TODO: Cases like '25to26' return 2526, which is not correct.
    """
    if any(month in string.lower() for month in ['january', 'february', 'march', 'april', 'may', 'june', 'july',
                                                 'august', 'september', 'october', 'november', 'december']):
        try:
            return dateparser.parse(string).timestamp().year
        except:  # Parse Error
            pass

    try:
        # If it is a word number (e.g. 'zero')
        numeric = w2n.word_to_num(string)
        return numeric
    except ValueError:
        pass

    # Remove any non-numeric characters except the decimal point and the negative sign
    string_re = re.sub("[^0-9\.\-]", "", string)

    if string_re.startswith('-'):
        string_re = '&' + string_re[1:]

    # Check if the string includes a range
    if "-" in string_re:
        # Split the string into parts based on the dash character
        parts = string_re.split("-")
        return coerce_to_numeric(parts[0].replace('&', '-'))
    else:
        string_re = string_re.replace('&', '-')

    try:
        # Convert the string to a float or int depending on whether it has a decimal point
        if "." in string_re:
            numeric = float(string_re)
        else:
            numeric = int(string_re)
    except:
        if no_string:
            raise ValueError
        # No numeric values. Return input
        return string
    return numeric


def object_detection_from_segmentation(img_name: str, object_name: str) -> List[Tuple[int, int, int, int]]:
    """
    Detects the coordinates of the object in the image using the segmentation mask.

    Parameters:
    - name of the image to retrieve the matching segmentation
    - name of the object to detect in the image
    
    Returns:
    - list of coordinates of the object in the image in the order (left, lower, right, upper)

    """
    img_name = img_name.replace(".jpg", "")
    # NOTE remove "interpolated" in file name when copying segmentations to data folder
    seg_img = Image.open(f"{config.dataset.data_path}/images/segmentations/{img_name}.png")
    unique_img_values = set(seg_img.getdata())
    coordinates = []
    np_image = np.array(seg_img)
    height = np_image.shape[0]

    # NOTE unrelated person is searched for as "person". This will trigger the search for all people in the image, which is not intended.
    # therefore removing unrelated person queries from dataset
    object_name = PRINTABLE_NAMES_TO_SEGMENTATIONS[object_name]

    if object_name in ["person", "people"]:
        # NOTE for now excluding unrelated people, as multiple people are not distinguisheable with this method
        people = ["patient", "anest", "circulator", "assistant_surgeon", "head_surgeon", "mps", "nurse"]
        label_values = [TRACK_TO_METAINFO[person]["label"] for person in people]
        people_mask = [label in unique_img_values for label in label_values]

        for label_value, is_selected in zip(label_values, people_mask):
            if is_selected:
                # Create a new image where only the query_value pixels are white, and all others are black
                img_processed = seg_img.point(lambda x: 255 if x == label_value else 0)
                bbox_coordinates = img_processed.getbbox()
                left, upper, right, lower = bbox_coordinates
                # Adjust the y-axis to match match ImagePatch coordinate system vs. PIL coordinate system
                adjusted_lower = height - upper
                adjusted_upper = height - lower
                # Ensure the coordinates are valid for cropping
                if adjusted_lower < adjusted_upper:
                    coordinates.append((left, adjusted_lower, right, adjusted_upper))
                else:
                    coordinates.append((left, adjusted_upper, right, adjusted_lower))  # Ensure lower < upper

    else:
        # retreive coordinates of only the specified object, which is a key in TRACK_TO_METAINFO
        label_value = TRACK_TO_METAINFO[object_name]["label"]
        if label_value in unique_img_values:
            # NOTE the coordinate system changes for numpy arrays
            # (0, 0)       (0, 1)       (0, 2)
            # (1, 0)       (1, 1)       (1, 2)
            # (2, 0)       (2, 1)       (2, 2)
            # this means that the x-axis is matching the x-axis of ImagePatch, but the y-axis is inverted
            matching_indices = np.argwhere(np_image == label_value)
            min_y, min_x = np.min(matching_indices, axis=0)
            max_y, max_x = np.max(matching_indices, axis=0)
            min_y_patch = height - max_y
            max_y_patch = height - min_y
            coordinates.append((min_x, min_y_patch, max_x, max_y_patch))

    return coordinates

def relation_from_ground_truth_SG(object_1, object_2, question, image_name):
    # check if there is a relation in question
    contained_relation = [relation for relation in PRINTABLE_NAMES_TO_RELATIONS.keys() if relation in question]
    if len(contained_relation) == 1:
        relation = contained_relation[0]
    else:
        relation = None

    # get correct relation label from FRAME_LOOKUP_TABLE
    gt_relation_frames = FRAMES_LOOKUP_TABLE
    azure_idx = image_name.split("-")[1].split(".")[0]

    # extract relation label index from lookup table
    # NOTE frames json does not contain a match for every azure index -> retrieve closest and second closest one
    closest_relation_idx = None
    second_closest_relation_idx = None
    min_difference = float("inf")
    second_min_difference = float("inf")

    for i, item in enumerate(gt_relation_frames):
        difference = abs(int(item[1]["azure"]) - int(azure_idx))
        if difference < min_difference:
            # Update second closest when a new closest is found
            second_min_difference = min_difference
            second_closest_relation_idx = closest_relation_idx
            # Update the closest
            min_difference = difference
            closest_relation_idx = i

        elif difference < second_min_difference:
            # Update the second closest when it's better than the current second but worse than the closest
            second_min_difference = difference
            second_closest_relation_idx = i

        if difference > second_min_difference:
            # Stop the loop when the difference starts increasing beyond the second smallest found
            break
    
    # case of non binary query about objects
    if relation is None:
        relation_labels = load_json(f"{config.dataset.data_path}/relation_labels/{str(closest_relation_idx).zfill(6)}.json")
        relations = relation_labels['rel_annotations']
        for relation_list in relations:
            if PRINTABLE_NAMES_TO_RELATION_OBJECTS[object_1] in relation_list and PRINTABLE_NAMES_TO_RELATION_OBJECTS[object_2] in relation_list:
                relation_string = f"the {RELATION_OBJECTS_TO_PRINTABLE_NAMES[relation_list[0]]} " \
                                    f"is {RELATIONS_TO_PRINTABLE_NAMES[relation_list[1]]} " \
                                    f"the {RELATION_OBJECTS_TO_PRINTABLE_NAMES[relation_list[2]]}"
                # in very few cases there are multiple relations between two objects, ground truth has only one however
                break
        return relation_string

    # case of binary query about objects
    else:
        # in case of using the closest frame (min_diff != 0), it can still be the wrong one
        # therefore check if the second closest frame contains the relation
        matches = []
        for index in [closest_relation_idx, second_closest_relation_idx]:
            relation_labels = load_json(f"{config.dataset.data_path}/relation_labels/{str(index).zfill(6)}.json")
            relations = relation_labels['rel_annotations']
            for relation_list in relations:
                if PRINTABLE_NAMES_TO_RELATION_OBJECTS[object_1] in relation_list and PRINTABLE_NAMES_TO_RELATION_OBJECTS[object_2] in relation_list:
                    if RELATIONS_TO_PRINTABLE_NAMES[relation_list[1]] == relation:
                        matches.append("yes")
        # return the relation as a match if it is found in either of the two closest frames
        return "yes" if "yes" in matches else "no"