from time import sleep
from src.model.image_classifier import BinaryImageClassifier


def main():
    positive = "real"
    positive_phrases = [
        "landscape photos",
        "nature photos",
        "real portrait people photos",
        "human being photos",
        "architecture photos",
        "cityscape photos",
        "wildlife photos",
        "human being photos",
    ]
    negative = "synthetic"
    negative_phrases = [
        "anime images",
        "abstract art ",
        "surreal artistic compositions",
        "illustrations",
        "surreal experimental designs",
        "fantasy images",
        "futuristic digital sci-fi images",
        "pixel art images",
        "digital retro images",
    ]
    train = BinaryImageClassifier(
        positive,
        negative,
        positive_phrases,
        negative_phrases,
        photos_per_phrase=2,
        batch_size=32,
    )
    train.train_()
    sleep(1)

    print("Testing lofi girl photo")
    train.predict(
        "/home/c_byrne/projects/ml/practical-deep-learning/creating-a-model/test-images/849779.jpg"
    )
    print("Testing Lady Bird Scene photo")
    train.predict(
        "/home/c_byrne/projects/ml/practical-deep-learning/creating-a-model/test-images/rgb-movie-scene-H533px_W800px-jpg-13.jpg"
    )
    print("Testing Windows XP Background photo")
    train.predict(
        "/home/c_byrne/projects/ml/practical-deep-learning/creating-a-model/test-images/rgb-background-real-day-H972px_W1474px-jpg-0.jpg"
    )


main()
