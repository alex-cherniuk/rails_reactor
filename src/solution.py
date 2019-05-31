import os
import sys
import numpy as np
import Image


def remove_duplicates(all_pairs):
    unique = []
    matched = []
    for i, (key11, key12) in enumerate(set(all_pairs)):
        if key11 == key12 or i in matched:
            continue
        if (key11, key12) not in unique and (key12, key11) not in unique:
            unique.append((key11, key12))
    return unique


class ImageAnalyser:
    __usage = "usage: solution.py [-h] --path PATH"

    @staticmethod
    def is_picture(path):
        return path.split(".")[-1] in ["jpg", "jpeg"]

    @staticmethod
    def keep_pictures_only(directory):
        return list(filter(ImageAnalyser.is_picture, os.listdir(directory)))

    @staticmethod
    def run(params: list):
        if len(params) == 1:
            print(ImageAnalyser.__usage + f"\n{params[0]}: error: the following arguments are required: --path")
        elif len(params) == 2 and params[1] == "--path":
            print(ImageAnalyser.__usage + f"\n{params[0]}: error: the following arguments are required: PATH")
        elif len(params) >= 2 and params[1] in ["-h", "--help"]:
            print(ImageAnalyser.__usage +
                  "\n\nFirst test task on images similarity." +
                  "\n\noptional arguments:" +
                  "\n  -h, --help            show this help message and exit" +
                  "\n  --path PATH           folder with images")
        elif len(params) == 3 and params[1] == "--path":
            directory = params[2]
            if not os.path.isdir(directory):
                print(ImageAnalyser.__usage + f"\n{params[0]}: error: directory does not exist")
            else:
                filenames = ImageAnalyser.keep_pictures_only(directory)
                if len(filenames) == 0:
                    print(ImageAnalyser.__usage + f"\n{params[0]}: directory is empty")
                else:
                    ImageAnalyser.analyse(filenames, directory)
        else:
            print(ImageAnalyser.__usage + f"\n{params[0]}: error: wrong usage format")

    @staticmethod
    def analyse(filenames, directory):
        images = {}
        for filename in filenames:
            image = Image.load_image(os.path.join(directory, filename))
            images[image.filename] = image

        duplicates_pairs = []
        similar_pairs = []
        modifications_pairs = []

        for i, key1 in enumerate(images.keys()):
            for key2 in images.keys():
                im1 = images[key1]
                im2 = images[key2]
                hash_similarity = np.where(im1.average_hash == im2.average_hash, 1, 0).sum() / 64
                if hash_similarity > 0.95 and im1.dimensions == im2.dimensions:
                    duplicates_pairs.append((key1, key2))
                elif hash_similarity > 0.95 and im1.dimensions != im2.dimensions:
                    modifications_pairs.append((key1, key2))
                elif hash_similarity > 0.85:
                    modifications_pairs.append((key1, key2))
                elif hash_similarity > 0.72:
                    similar_pairs.append((key1, key2))

        duplicates_pairs = remove_duplicates(duplicates_pairs)
        similar_pairs = remove_duplicates(similar_pairs)
        modifications_pairs = remove_duplicates(modifications_pairs)

        print("*****Duplicates:*****")
        for (key1, key2) in duplicates_pairs:
            print(key1, key2)

        print("*****Similar*****")
        for (key1, key2) in similar_pairs:
            print(key1, key2)

        print("*****Modifications*****")
        for (key1, key2) in modifications_pairs:
            print(key1, key2)


if __name__ == "__main__":
    ImageAnalyser.run(sys.argv)
