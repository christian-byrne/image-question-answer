import os

from simple_image_download import simple_image_download as simp
from logging_.log_and_print import Logger

from typing import List


class GoogleImageScraper:
    def __init__(self):
        self.logger = Logger("GoogleImageScraper")
        self.path = os.path.dirname(os.path.abspath(__file__))
        self.relpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    def scrape_images(self, keywords: str, limit: int) -> List[str]:
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
