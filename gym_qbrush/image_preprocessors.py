from PIL import Image


def greyscale(image):
    return image.convert('L')


def rgb(image):
    return image.convert('RGB')


def resize(new_size):
    def f_resize(image):
        return image.resize(new_size, Image.ANTIALIAS)
    return f_resize
