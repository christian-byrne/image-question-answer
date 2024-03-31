import os
from simple_image_download import simple_image_download as simp
from fastcore.all import *
from duckduckgo_search import DDGS
from fastdownload import download_url
from typing import List, Tuple
import time

from logging_.log_and_print import Logger


class GoogleImageScraper:
    def __init__(self):
        self.logger = Logger("GoogleImageScraper")
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.relpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def scrape_images(self, keywords, limit):
        response = simp.simple_image_download()
        response.download(keywords, limit)
        dl_path = os.path.join(self.path, "simple_images", keywords)

        self.logger.log(
            f"Images downloaded successfully to {self.relpath}/simple_images/{keywords}"
        )

        dl_paths = []
        for img in os.listdir(dl_path):
            dl_paths.append(os.path.join(dl_path, img))
        return dl_paths


class DuckDuckGoImageScraper:
    def __init__(self):
        self.logger = Logger("DuckDuckGoImageScraper")

    def scrape(self, category: str, search_phrases: List[Tuple[str, int]]) -> List[dict]:
        """
        Scrapes images from the web based on the given search phrases.

        Args:
            search_phrases (List[Tuple[str, int]]): A list of tuples containing the search phrases and the limit of images to scrape for each phrase.

        Returns:
            List[dict]: A list of dictionaries representing the scraped images.
        """
        ret = []
        for phrase, limit in search_phrases:
            real_photos = self.__scrape_images(category, phrase, limit)
            ret += real_photos
            time.sleep(10)
        return ret

    def _search_images(self, search_phrase, max_images=30):
        with DDGS() as ddg:
            results = ddg.images(keywords=search_phrase)
            # Only return up to max_images
            truncated = results[:max_images]
            return L(truncated)

    def __scrape_images(self, category: str, search_phrase: str, max_images: int = 30) -> List[dict]:
        """
        Scrapes images from the web based on the given search phrase.

        Args:
            search_phrase (str): The phrase to search for images.
            max_images (int, optional): The maximum number of images to scrape. Defaults to 30.

        Returns:
            List[{
                "title": (str) Title of the image,
                "image": (str) URL of the image,
                "thumbnail": (str) URL of the thumbnail,
                "url": (str) URL of the page containing the image,
                "height": (int) Height of the image,
                "width": (int) Width of the image,
                "source": (str) Website source of the image,
                "download_path": (Path) Path to the downloaded image
            }]: A list of dictionaries containing image metadata.
            }

        """
        image_results = self._search_images(search_phrase, max_images)
        download_path = os.path.join("duckduckgo_image_downloads", search_phrase)

        for index, result in enumerate(image_results):
            parsed_ext = Path(result["image"]).suffix
            filename = f"{search_phrase}_{index}{parsed_ext}"
            self.logger.log(f"Downloading image {index + 1} of {len(image_results)}\n")
            for k, v in result.items():
                self.logger.log(f"{k}: {v}")

            dl_path = os.path.join(category, download_path, filename)
            try:
                res_dl_path = download_url(
                    result["image"], dl_path, timeout=10, show_progress=False
                )
            except Exception as e:
                self.logger.log(
                    f"Failed to download image {index + 1} of {len(image_results)}\n"
                )
                self.logger.log(f"Error: {e}")
                continue
            self.logger.log(f"Downloaded image to {res_dl_path}")

            result["download_path"] = dl_path

        self.logger.log(
            f"Images downloaded successfully to duckduckgo_images/{search_phrase}"
        )

        return image_results
