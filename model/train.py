import time
from pathlib import Path
import os
from scrapers.images.web_image_scrape import DuckDuckGoImageScraper
from fastai.vision.all import (
    DataBlock,
    ImageBlock,
    CategoryBlock,
    get_image_files,
    RandomSplitter,
    parent_label,
)
from fastai.vision.all import Resize, vision_learner, error_rate, resnet18, Learner, PILImage

from logging_.log_and_print import Logger

from typing import List, Tuple


from PIL import Image


class TrainClassifier:
    def __init__(
        self,
        positive,
        negative,
        positive_phrases,
        negative_phrases,
        photos_per_phrase=2,
        batch_size=32,
    ):
        self.logger = Logger("TrainClassifier")
        self.positive = positive
        self.negative = negative
        self.photos_per_phrase = photos_per_phrase
        self.positive_phrases = [
            (phrase, photos_per_phrase) for phrase in positive_phrases
        ]
        self.negative_phrases = [
            (phrase, photos_per_phrase) for phrase in negative_phrases
        ]

        self.batch_size = batch_size

        self.abs_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.images_path = Path(
            os.path.join(self.abs_path, "duckduckgo_image_downloads")
        )

        self.positive_training_photos, self.negative_training_photos = self.get_photos()
        self.logger.log(f"Positive training photos: {self.positive_training_photos}")
        self.logger.log(f"Negative training photos: {self.negative_training_photos}\n")

        self.test_images_path()
        self.create_datablock()




    def predict(self, image_path) -> str:
        prediction, _, probs = self.model.predict(PILImage.create(image_path))
        print(f"This is a {prediction} image")
        print(f"Probability it's a {self.positive} image: {probs[0]:.4f}")

    def train_(self):
        learn_ = vision_learner(self.data_loader, resnet18, metrics=error_rate)

        # Verify the implementation of the vision_learner function constructs and returns a Learner object
        assert (
            learn_ is not None
        ), "The vision_learner function did not return a Learner object"
        assert isinstance(
            learn_, Learner
        ), "The vision_learner function did not return a Learner object"

        learn_.fine_tune(3)

        self.model = learn_

    def create_datablock(self):
        self.logger.log("Creating datablock")
        data = DataBlock(
            blocks=(ImageBlock, CategoryBlock),
            get_items=get_image_files,
            splitter=RandomSplitter(valid_pct=0.2, seed=42),
            get_y=parent_label,
            item_tfms=[Resize(128, method="squish")],
        ).dataloaders(self.images_path, bs=self.batch_size)

        self.data_loader = data

        self.data_loader.show_batch()

    def test_images_path(self):
        res = get_image_files(self.images_path)
        if len(res) == 0:
            raise FileNotFoundError(
                f"No images found in the self.images_path: {self.images_path}"
            )

        tests_ct = 0
        for img in res:
            img = Image.open(img)
            if img.verify() is None:
                tests_ct += 1
            if tests_ct > 10:
                break

    def get_photos(self):
        # Check if images already exist
        self.logger.log("Checking if images already exist")
        images_path = "duckduckgo_image_downloads"
        positive_path = os.path.join(images_path, self.positive)
        negative_path = os.path.join(images_path, self.negative)
        if os.path.exists(positive_path) and os.path.exists(negative_path):
            if (
                len(os.listdir(positive_path)) >= self.batch_size
                and len(os.listdir(negative_path)) >= self.batch_size
            ):
                self.logger.log("Images already exist")
                # Paths to all images in positive directory and subdirectories
                positive_photos_recurse = [
                    os.path.join(dp, f)
                    for dp, dn, filenames in os.walk(positive_path)
                    for f in filenames
                ]
                # Paths to all images in negative directory and subdirectories
                negative_photos_recurse = [
                    os.path.join(dp, f)
                    for dp, dn, filenames in os.walk(negative_path)
                    for f in filenames
                ]
                return positive_photos_recurse, negative_photos_recurse

        else:
            raise FileNotFoundError("Images not found")

        # Scrape images if they don't exist
        self.logger.log("Scraping images")
        image_scraper = DuckDuckGoImageScraper()
        positive_photos = image_scraper.scrape(self.positive, self.positive_phrases)
        negative_photos = image_scraper.scrape(self.negative, self.negative_phrases)
        return [photo["download_path"] for photo in positive_photos], [
            photo["download_path"] for photo in negative_photos
        ]
