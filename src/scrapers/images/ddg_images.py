import os
from pathlib import Path
import time

from duckduckgo_search import DDGS
from fastdownload import download_url
from fastcore.all import *

from logging_.log_and_print import Logger
from utils.path_utils import ProjPaths

from typing import List, Tuple


class DuckDuckGoImageScraper:
    def __init__(
        self,
        downloads_dirname: str = "duckduckgo_image_downloads",
        sleep_interval: float = 10.0,
        timout: int = 10,
    ):
        self.logger = Logger("DuckDuckGoImageScraper")
        self.dl_dirname = downloads_dirname
        self.dl_path = ProjPaths.get_data(self.dl_dirname)

        self.sleep_interval = sleep_interval
        self.timeout = timout

        self.logger.log(f"Download path (abs): {self.dl_path}")
        self.logger.log(f"Download path (rel): {self.dl_dirname}")

    def get_dl_path(self) -> Path:
        return self.dl_path

    def scrape(
        self, category: str, search_phrases: List[Tuple[str, int]]
    ) -> List[dict]:
        """
        Scrapes images from the web based on the given search phrases.

        Args:
            search_phrases (List[Tuple[str, int]]): A list of tuples containing the search phrases and the limit of images to scrape for each phrase.

        Returns:
            List[dict]: [{
                "title": (str) Title of the image,
                "image": (str) URL of the image,
                "thumbnail": (str) URL of the thumbnail,
                "url": (str) URL of the page containing the image,
                "height": (int) Height of the image,
                "width": (int) Width of the image,
                "source": (str) Website source of the image,
                "download_path": (Path) Path to the downloaded image
            },...]
        """
        ret = []
        for phrase, limit in search_phrases:
            downloaded_photos = self.__search_scrape_images(category, phrase, limit)
            ret += downloaded_photos
            time.sleep(self.sleep_interval)
        return ret

    def search_images(self, search_phrase: str, max_images: int = 30) -> List[dict]:
        with DDGS() as ddg:
            results = ddg.images(keywords=search_phrase)
            # Only return up to max_images
            truncated = results[:max_images]
            return L(truncated)

    def __search_scrape_images(
        self, category: str, search_phrase: str, max_images: int = 30
    ) -> List[dict]:
        image_results = self.search_images(search_phrase, max_images)

        for index, result in enumerate(image_results):
            filename = f"{search_phrase}_{index}{Path(result['image']).suffix}"
            self.logger.log(f"Downloading image {index + 1} of {len(image_results)}\n")
            self.logger.log(result)

            photo_dl_path = ProjPaths.get_data(
                self.dl_path / category / search_phrase / filename
            )

            try:
                res_dl_path = download_url(
                    result["image"], photo_dl_path, timeout=self.timeout, show_progress=False
                )
            except Exception as e:
                self.logger.log(
                    f"Failed to download image {index + 1} of {len(image_results)}\n",
                    color_override="red",
                )
                self.logger.log(f"Error: {e}", color_override="red")
                continue

            self.logger.log(f"Downloaded image to {res_dl_path}")
            result["download_path"] = photo_dl_path

        self.logger.log(
            f"Images downloaded successfully to {self.dl_path / category / search_phrase}"
        )

        return image_results
