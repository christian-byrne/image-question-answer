"""https://docs.fast.ai/learner.html#Learner.predict"""

from pathlib import Path

from PIL import Image
from fastai.vision.all import (
    DataBlock,
    ImageBlock,
    CategoryBlock,
    get_image_files,
    RandomSplitter,
    parent_label,
)
from fastai.vision.all import (
    Resize,
    vision_learner,
    error_rate,
    resnet18,
    Learner,
    PILImage,
)

from utils.path_utils import ProjPaths
from scrapers.images.ddg_images import DuckDuckGoImageScraper
from constants import PHOTO_DL_DIRNAME, PICTURE_EXTENSION_LIST
from logging_.log_and_print import Logger

from typing import List, Tuple, Union, Any


class BinaryImageClassifier:
    def __init__(
        self,
        positive,
        negative,
        positive_phrases,
        negative_phrases,
        photos_per_phrase=4,
        batch_size=32,
        img_res=128,
    ):
        self.logger = Logger("BinaryImageClassifier", "blue")
        self.positive = positive
        self.negative = negative
        self.batch_size = batch_size
        self.img_res = img_res
        self.__set_phrases(photos_per_phrase, positive_phrases, negative_phrases)

        self.image_scraper = DuckDuckGoImageScraper(PHOTO_DL_DIRNAME)
        self.training_images_path = self.image_scraper.get_dl_path()
        self.positive_path = self.training_images_path / self.positive
        self.negative_path = self.training_images_path / self.negative
        self.positive_training_photos, self.negative_training_photos = self.get_photos()

        self.__verify_dataset()
        self.__create_datablock()
        self.logger.log(self)

    def __str__(self):
        return "\n" + "\n".join(
            [
                f"BinaryImageClassifier(positive={self.positive}",
                f"negative={self.negative}",
                f"photos_per_phrase={self.photos_per_phrase}",
                f"batch_size={self.batch_size})",
                f"Positive training photos preview: {[str(photo)[-36:] for photo in self.positive_training_photos[:4]]}",
                f"Negative training photos preview: {[str(photo)[-36:] for photo in self.negative_training_photos[:4]]}",
            ]
        )

    def __len__(self):
        return len(self.positive_training_photos) + len(self.negative_training_photos)

    def predict(
        self, image_path: Path
    ) -> tuple[Any | None, Any | None, Any, Any] | tuple[Any | None, Any, Any]:
        """https://docs.fast.ai/learner.html"""
        prediction, decoded_prediction, probs = self.model.predict(
            PILImage.create(image_path)
        )
        return prediction, decoded_prediction, probs

    def print_prediction(
        self,
        predict_out: (
            tuple[Any | None, Any | None, Any, Any] | tuple[Any | None, Any, Any]
        ),
    ) -> None:
        prediction, _, probs = predict_out
        if probs[0] * 100 < 1:
            prob_string = f"<1% ({probs[0]:.4f})"
        else:
            prob_string = f"{probs[0] * 100:.1f}% ({probs[0]:.4f})"
        prediction_string = f"{prediction}"

        print(f"{'Prediction:':<42}{prediction_string:>38}")
        print(
            f"{'Probability it is a ' + self.positive + ' image:':<42}{prob_string:>38}"
        )

    def train_(self, epochs: int = 4) -> None:
        learn_ = vision_learner(self.data_loader, resnet18, metrics=error_rate)

        # Verify the implementation of the vision_learner function constructs and returns a Learner object
        assert (
            learn_ is not None
        ), "The vision_learner function did not return a Learner object"
        assert isinstance(
            learn_, Learner
        ), "The vision_learner function did not return a Learner object"

        self.logger.log("Training model")
        print()

        learn_.fine_tune(epochs)
        self.model = learn_

    def collect_images_recursive(self, path: Path) -> List[Path]:
        ret = []
        for item in path.iterdir():
            if item.is_dir():
                ret += self.collect_images_recursive(item)
            elif item.is_file() and item.suffix in PICTURE_EXTENSION_LIST:
                ret.append(item)

        return ret

    def get_photos(self) -> Tuple[List[Path], List[Path]]:
        # Check if images already exist
        self.logger.log("Checking if images exist")
        existing_photos = self.__get_photos_if_exist()
        if existing_photos:
            self.logger.log(
                f"Images already exist: {len(existing_photos[0])} positive photos, {len(existing_photos[1])} negative photos"
            )
            return existing_photos

        # Scrape images if they don't exist
        self.logger.log("Scraping images")
        positive_photos = self.image_scraper.scrape(
            self.positive, self.positive_phrases
        )
        negative_photos = self.image_scraper.scrape(
            self.negative, self.negative_phrases
        )

        return [photo["download_path"] for photo in positive_photos], [
            photo["download_path"] for photo in negative_photos
        ]

    def __verify_dataset(self) -> None:
        res = get_image_files(self.training_images_path)
        if len(res) == 0:
            raise FileNotFoundError(
                f"No images found in the self.images_path: {self.training_images_path}"
            )

        tests_ct = 0
        max_tests = min(self.batch_size // 2, 16)
        for img in res:
            try:
                img = Image.open(img)
                if img.verify() is None:
                    tests_ct += 1
            except Exception as e:
                self.logger.log(
                    f"[DataBlock Verification] Failed to verify image {img}"
                )
                self.logger.log(f"Error: {e}")
            if tests_ct >= max_tests:
                break

    def __create_datablock(self) -> None:
        self.logger.log("Creating datablock")
        data = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            splitter=RandomSplitter(valid_pct=0.2, seed=42),
            get_y=parent_label,
            item_tfms=[Resize(self.img_res, method="squish")],
        ).dataloaders(self.training_images_path, bs=self.batch_size)

        self.data_loader = data

        self.data_loader.show_batch()

    def __get_photos_if_exist(self) -> Union[Tuple[List[Path], List[Path]], bool]:
        pos_photos = self.collect_images_recursive(self.positive_path)
        neg_photos = self.collect_images_recursive(self.negative_path)
        if (
            self.positive_path.exists()
            and len(pos_photos) >= self.batch_size
            and self.negative_path.exists()
            and len(neg_photos) >= self.batch_size
        ):
            return pos_photos, neg_photos
        return False

    def __set_phrases(
        self,
        photos_per_phrase: int,
        positive_phrases: List[Tuple[str, int]],
        negative_phrases: List[Tuple[str, int]],
    ) -> None:
        required_photos_per_phrase = (
            self.batch_size // min(len(positive_phrases), len(negative_phrases))
        ) + 1
        if photos_per_phrase < required_photos_per_phrase:
            self.logger.log(
                f"Number of photos per phrase is too low to create batches of size {self.batch_size}."
                + f"Increasing to {required_photos_per_phrase}"
            )
            self.photos_per_phrase = required_photos_per_phrase
        else:
            self.photos_per_phrase = photos_per_phrase

        self.positive_phrases = [
            (phrase, photos_per_phrase) for phrase in positive_phrases
        ]
        self.negative_phrases = [
            (phrase, photos_per_phrase) for phrase in negative_phrases
        ]
