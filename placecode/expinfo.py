import json
import os
import errno
from uuid import uuid4
import warnings


class ParameterNotFoundError(Exception):
    pass


class ExpInfo():
    # TODO add proper documentation
    """
    Attributes:

    Required:
    fpath_info: str, the file path of the experiment info json file
    fpath_caim: str, the file path of the CaImAn data (hdf5 file)
    fpath_loco: str, the file path of the locomotion data (hdf5 file)
    mouse_ID: str, mouse ID
    condition: str, condition

    Optional:
    uuid: str, the hexadecimal form of the experiment identifier
    exp_type: str, the experiment type
    fpath_lfp: str, the file path of the LFP file (abf file)
    """

    def __init__(self, fpath_json: str):
        """Create an instance of ExpInfo by passing the file path of the 
        experiment-specific json file.
        Parameters
        ----------
        json_fpath : str
            the file path to the json file. 
            Examples: 
              os.path.join(folder, fname),
              os.path.normpath("C:/a/b/c.txt")
        Raises
        ------
        FileNotFoundError
            raised if the specified file does not exist.
        ValueError
            raised if the specified file is not a json file.
        ParameterNotFoundError
            raised if one of the following keys not found in the json file:
                mouse_ID, condition, home_folder

        """
        if not isinstance(fpath_json, str):
            raise TypeError(
                f"Expected json_fpath argument to be str, received {type(fpath_json)}: {fpath_json}")
        if not os.path.exists(fpath_json):
            raise FileNotFoundError(errno.ENOENT, os.strerror(
                errno.ENOENT), f"File {fpath_json} does not exist.")
        if not os.path.splitext(fpath_json)[-1] == ".json":
            raise ValueError(f"File {fpath_json} is not a json file.")
        with open(fpath_json, "r") as f:
            dict_json = json.load(f)
        self.fpath_info = fpath_json
        # try to read required parameters first
        if "mouse_ID" in dict_json:
            self.mouse_ID = dict_json["mouse_ID"]
        else:
            raise ParameterNotFoundError(
                f"required parameter mouse_ID not found in json file {fpath_json}")

        if "condition" in dict_json:
            self.condition = dict_json["condition"]
        else:
            raise ParameterNotFoundError(
                f"parameter condition not found in json file {fpath_json}")

        if "fpath_caim" in dict_json:
            self.fpath_caim = dict_json["fpath_caim"]
        else:
            raise ParameterNotFoundError(
                f"parameter fpath_caim (file path of CaImAn hdf5 data) not found in json file {fpath_json}")

        if "fpath_loco" in dict_json:
            self.fpath_loco = dict_json["fpath_loco"]
        else:
            raise ParameterNotFoundError(
                f"parameter fpath_loco (locomotion hdf5 data) not found in json file {fpath_json}")
        # read optional parameters
        if "uuid" in dict_json:
            self.uuid = dict_json["uuid"]
        else:
            self.uuid = uuid4().hex
            warnings.warn(
                f"No uuid found for experiment. Assigning {self.uuid}")
        if "exp_type" in dict_json:
            self.exp_type = dict_json["exp_type"]
        else:
            self.exp_type = None
        if "fpath_lfp" in dict_json:
            self.fpath_lfp = dict_json["fpath_lfp"]
        else:
            self.fpath_lfp = None
