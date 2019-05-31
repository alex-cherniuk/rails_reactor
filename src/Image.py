import os
import numpy as np
from PIL import Image as Img
import matplotlib.pyplot as plt


class Image:
    def __init__(self, image, dimensions, path=None):
        self.__image = image
        (self.H, self.W) = dimensions
        self.dimensions = dimensions
        self.path = path
        if path is not None:
            self.filename = path.split("\\")[-1]
            self.dtype = self.filename.split(".")[-1]
        self.__hashed = None
        self.parent = None
        self.duplicates = []
        self.modifications = []
        self.similar = []

    @property
    def average_hash(self):
        if self.__hashed is None:
            self.__hashed = self.get_average_hash(self.__image)
        return self.__hashed

    @staticmethod
    def get_average_hash(image, hash_size=8):
        if hash_size < 2:
            raise ValueError("Hash size must be greater than or equal to 2")

        # reduce size
        image = resize_image(image, shape=(hash_size, hash_size))

        # find average pixel value; 'image' is an array of the pixel values, ranging from 0 (black) to 255 (white)
        avg = image.mean()

        # create string of bits
        diff = image > avg
        return diff

    @staticmethod
    def binary_array_to_hex(array):
        """
        internal function to make a hex string out of a binary array.
        """
        bit_string = ''.join(str(b) for b in 1 * array.flatten())
        width = int(np.ceil(len(bit_string) / 4))
        return '{:0>{width}x}'.format(int(bit_string, 2), width=width)

    def get_image(self):
        return self.__image

    def set_image(self, image):
        image = np.asarray(image, dtype=float)
        if len(image.shape) == 2:
            self.__image = image
            (self.H, self.W) = self.__image.shape
        else:
            print("Assignment Error. Given input is not an image")

    def convolve(self, mask):
        mask = np.asarray(mask, dtype=float)
        if len(mask.shape) != 2:
            print("Invalid Mask. Please input a 2D Mask")
            return
        (h, w) = mask.shape
        pad_height = (h-1) // 2
        pad_width = (w-1) // 2
        image = np.ones((self.H+pad_height*2, self.W+pad_width*2))*128
        new_img = np.ones((self.H+pad_height*2, self.W+pad_width*2))
        image[pad_height:-pad_height, pad_width:-pad_width] = self.__image

        for i in range(pad_height, self.H+pad_height):
            for j in range(pad_width, self.W+pad_width):
                new_img[i, j] = sum(sum(image[i-pad_height:i+h-pad_height, j-pad_width:j+w-pad_width]*mask))

        return new_img[pad_height:-pad_height, pad_width:-pad_width]


def resize_image(image, shape=(8, 8)):
    """
    Resize image to new image with shape (width, height)
    :param image: ndarray with original shape
    :param shape: output array's shape (list, tuple with length 2)
    :return: ndarray with shape (width, height)
    """
    (old_height, old_width) = image.shape
    (height, width) = shape
    resized_image = np.zeros((height, width))
    for h in range(height):
        for w in range(width):
            new_width = int(w * old_width / width)
            new_height = int(h * old_height / height)
            resized_image[h][w] = image[new_height][new_width]
    return resized_image


def load_image(path: str, mode: str = 'L', resize: bool = True, shape=(128, 128)):
    try:
        image = Img.open(os.path.abspath(path)).convert(mode)
    except Exception:
        print("Error! Could not read the image from the path specified: %s" % os.path.abspath(path))
        return
    image = np.asarray(image, dtype=float)
    if resize:
        return Image(resize_image(image, shape), shape, path)
    return Image(image, image.shape, path)


if __name__ == "__main__":
    image_path1 = "./images/1_duplicate.jpg"
    image_path2 = "./images/1.jpg"
    img1 = load_image(image_path1)
    img2 = load_image(image_path2)

    print(np.where(img1.average_hash == img2.average_hash, 1, 0).sum()/64.)

    plt.imshow(img1.get_image(), cmap="gray")
    print(Image.binary_array_to_hex(img1.average_hash))

    kernel_3x3_1 = np.array([[0, -1,  0],
                             [-1,  4, -1],
                             [0, -1,  0]])

    new_im = img1.convolve(kernel_3x3_1)
    plt.imshow(new_im, cmap="gray")
    print(new_im.var())
    plt.show()
