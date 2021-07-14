import os
import sys
from typing import Generator


def filter_by_type(directory: str, extension: str)-> Generator:
    """Filter files in a directory by file extension.

    :param directory: files whose extension match in this directory
    :param extension: [1] elem of os.path.splitext, `.jpg`
    :return: generator whose elements are type `DirEntry`.
    """
    if not os.path.exists(directory):
        sys.exit(f'Not exist the directory: {directory}')

    return (f for f in os.scandir(directory) if os.path.splitext(f)[1] == extension)
