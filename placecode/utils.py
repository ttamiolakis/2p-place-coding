from tkinter import Tk  # use tkinter to open files
from tkinter.filedialog import askopenfilename, askdirectory
import os.path


def raise_above_all(window):
    window.attributes('-topmost', 1)
    window.attributes('-topmost', 0)


def open_file(title: str = "Select file") -> str:
    """Opens a tkinter dialog to select a file. Returns the path of the file.

    Parameters
    ----------
    title : str, optional
        The message to display in the open directory dialog, by default "Select file".
    :return: the absolute path of the directory selected.

    Returns
    -------
    str
        The absolute path of the file selected, or "." if Cancel was pressed.
    """
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()  # keep root window from appearing
    return os.path.normpath(askopenfilename(title=title))


def open_dir(title: str = "Select data directory", ending_slash: bool = False) -> str:
    """Opens a tkinter dialog to select a folder. Returns the path of the folder.

    Parameters
    ----------
    title : str, optional
        The message to display in the open directory dialog, by default "Select data directory"
    ending_slash : bool, optional
        _description_, by default False

    Returns
    -------
    str
        _description_
    """
    """

    :param title: 
    :return: the absolute path of the directory selected.
    """
    root = Tk()
    root.attributes("-topmost", True)
    root.withdraw()  # keep root window from appearing
    folder_path = askdirectory(title=title)
    if ending_slash:
        folder_path += "/"
    return os.path.normpath(folder_path)
