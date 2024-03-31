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

    testing_rate = 20
    testing_batch_size = 64
    for i in range(min(EXAMPLE_COUNT, len(examples))):
        example = examples[i]
        print(f"\n\nExample {i + 1}: {example['title']}\n")
        print(colored(f"{'Positive':<42}{example['positive']:>38}", "light_green"))
        print(colored(f"{'Negative':<42}{example['negative']:>38}", "light_red"))
        print("\nPositive Image Search Phrases:")
        for phrase in example["positive_phrases"]:
            print(f"  - ~{testing_rate} images from searching: '{phrase}'")
        print("\nNegative Image Search Phrases:")
        for phrase in example["negative_phrases"]:
            print(f"  - ~{testing_rate} images from searching: '{phrase}'")
        print()

        model = BinaryImageClassifier(
            example["positive"],
            example["negative"],
            example["positive_phrases"],
            example["negative_phrases"],
            photos_per_phrase=testing_rate,
            batch_size=testing_batch_size,
        )
        model.train_(epochs=4)
        sleep(1)

        for test_photo in example["test_photos"]:
            print()
            print(f"{'Testing Photo:':<42}{test_photo[0]:>38}")
            print(f"{'Expected Result:':<42}{test_photo[2]:>38}")
            res = model.predict(
                ProjPaths.get_data(Path(f"test_images/{test_photo[1]}"))
            )
            model.print_prediction(res)
            if res[0].lower() == test_photo[2].lower():
                print((" ✅✅ " + colored("SUCESS", "green")) * 3)
            else:
                print((" ❌❌ " + colored("FAILURE", "red")) * 3)


if __name__ == "__main__":
    main()
    print()
