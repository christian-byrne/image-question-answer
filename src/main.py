from time import sleep
from pathlib import Path

from model.image_classifier import BinaryImageClassifier
from utils.path_utils import ProjPaths

from termcolor import colored


def main():
    EXAMPLE_COUNT = 1
    examples = [
        {
            "title": "Determine if a photo is real or synthetic",
            "positive": "real",
            "negative": "synthetic",
            "positive_phrases": [
                "landscape photos",
                "nature photos",
                "real portrait people photos",
                "everyday person photos",
                "architecture photos",
                "cityscape photos",
                "wildlife photos",
                "human being photos",
            ],
            "negative_phrases": [
                "anime images",
                "abstract art ",
                "surreal artistic compositions",
                "illustrations",
                "surreal experimental designs",
                "fantasy images",
                "futuristic digital sci-fi images",
                "pixel art images",
                "digital retro images",
            ],
            "test_photos": [
                ("Lofi Girl", "849779.jpg", "synthetic"),
                ("Lady Bird Scene", "rgb-movie-scene-H533px_W800px-jpg-13.jpg", "real"),
                (
                    "Windows XP Background",
                    "rgb-background-real-day-H972px_W1474px-jpg-0.jpg",
                    "real",
                ),
            ],
        }
    ]

    for i in range(min(EXAMPLE_COUNT, len(examples))):
        example = examples[i]
        print(f"Example {i + 1}: {example['title']}\n")

        model = BinaryImageClassifier(
            example["positive"],
            example["negative"],
            example["positive_phrases"],
            example["negative_phrases"],
            photos_per_phrase=4,
            batch_size=16,
        )
        model.train_()
        sleep(1)

        for test_photo in example["test_photos"]:

            print(f"Testing {test_photo[0]} photo")
            res = model.predict(
                ProjPaths.get_data(Path(f"test_images/{test_photo[1]}"))
            )
            print(f"Expected: {test_photo[2]}")
            if res[0].lower() == test_photo[2].lower():
                print("✅✅" + colored("SUCESS", "green"))
            else:
                print("❌❌" + colored("~~FAILURE~~", "red"))

            print("\n")


if __name__ == "__main__":
    main()
