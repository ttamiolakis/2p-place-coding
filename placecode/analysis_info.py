import errno
import json
from math import ceil
import os
import warnings
from uuid import uuid4
from typing import Dict


class ParameterNotFoundError(Exception):
    pass


class ExpInfo():
    # TODO add proper documentation to attributes
    # TODO: some parameters (belt_length_mm, for example) can be read out from the Matlab matching parameters
    """
    Attributes:

    Required:
    fpath_info: str, the file path of the experiment info json file
    fpath_caim: str, the file path of the CaImAn data (hdf5 file)
    fpath_loco: str, the file path of the locomotion data (hdf5 file)
    mouse_ID: str, mouse ID
    condition: str, condition
    belt_length_mm: float, the length of the belt in mm

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
        # TODO: typecheck the parameters!
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

        if "belt_length_mm" in dict_json:
            self.belt_length_mm = dict_json["belt_length_mm"]
        else:
            raise ParameterNotFoundError(
                f"parameter belt_length_mm (belt total length in mm) not found in json file {fpath_json}")

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


class AnalysisParams():
    """
    Attributes:

    Required:
    * peak_threshold: float, the required height for a peak to not be rejected. Used as scipy.signal.find_peaks height parameter
    * peak_distance: number, required minimum distance (in samples unit) between peaks. Used as scipy.signal.find_peaks distance parameter
    * n_events_threshold: int, threshold on how many events a cell should have to be included in the analysis. It is an exclusive threshold, i.e. the true event count must exceed the threshold.
    * n_shuffle: int, the number of shuffling of spiking events to take place.
    * At least one of
        * n_bins: int, the desired number of spatial bins. Calculated from ExpInfo.belt_length_mm and bin_size 
            if bin_size is given. If both n_bins and bin_size given, uses n_bins.
        * bin_size: float, the desired spatial bin size in mm. Calculated from ExpInfo.belt_length_mm and n_bins 
            if n_bins is given. If both n_bins and bin_size given, uses n_bins.

    Generated:
    * rounds_included: np.array(shape=(n_rounds,), dtype=np.int8): an array of binary flags. 1 if round was included in analysis, 0 if discarded.
    * n_units: int, the number of cells in the recording

    """

    def __init__(self, fpath_json: str):
        """Read the contents of the json file.

        Parameters
        ----------
        fpath_json : str
            The path to the json file containing the analysis parameters. See class documentation on which parameters it should contain.
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
        # try to read required parameters first
        # TODO: typecheck the parameters
        if "peak_threshold" in dict_json:
            self.peak_threshold = float(dict_json["peak_threshold"])
        else:
            raise ParameterNotFoundError(
                f"required parameter peak_threshold not found in json file {fpath_json}")
        if "peak_distance" in dict_json:
            self.peak_distance = float(
                dict_json["peak_distance"])  # TODO: float or int?
        else:
            raise ParameterNotFoundError(
                f"required parameter peak_distance not found in json file {fpath_json}")
        if "n_events_threshold" in dict_json:
            self.n_events_threshold = float(
                dict_json["n_events_threshold"])  # TODO: float or int?
        else:
            raise ParameterNotFoundError(
                f"required parameter n_events_threshold not found in json file {fpath_json}")
        # get either n_bins or bin_size. Prioritize n_bins if both given.
        if "n_bins" in dict_json:
            self.n_bins = int(dict_json["n_bins"])
            self.bin_size = None
        elif "bin_size" in dict_json:  # no n_bins provided, try to infer
            self.bin_size = dict_json["bin_size"]
            self.n_bins = None
        else:  # neither n_bins nor bin_size is given
            raise ParameterNotFoundError(
                "Either n_bins or bin_size should be given (for spatial binning)!")
        if "n_shuffle" in dict_json:
            self.n_shuffle = dict_json["n_shuffle"]
        else:
            raise ParameterNotFoundError(
                f"required parameter n_shuffle not found in json file {fpath_json}")
        # Initialize generated parameters
        self.rounds_included = None
        self.n_units = None

    def to_dict(self) -> Dict:
        """
        Returns the attributes as a dict.
        """
        dict_attrs = dict()
        dict_attrs["peak_threshold"] = self.peak_threshold
        dict_attrs["peak_distance"] = self.peak_distance
        dict_attrs["n_events_threshold"] = self.n_events_threshold
        dict_attrs["n_shuffle"] = self.n_shuffle
        dict_attrs["n_bins"] = self.n_bins
        dict_attrs["bin_size"] = self.bin_size
        dict_attrs["rounds_included"] = self.rounds_included
        dict_attrs["n_units"] = self.n_units
        dict_attrs["peak_threshold"] = self.peak_threshold
        return dict_attrs

    def read_exp_info(self, exp_info: ExpInfo) -> None:
        """Read out relevant experiment info from an ExpInfo object.

        Parameters
        ----------
        exp_info : ExpInfo
            The ExpInfo object of the experiment.
        """
        self.belt_length_mm = exp_info.belt_length_mm
        self.condition = exp_info.condition
        self.mouse_ID = exp_info.mouse_ID
        if self.n_bins is not None:
            self.bin_size = self.belt_length_mm / self.n_bins
            if self.belt_length_mm % self.n_bins != 0:
                self.bin_size += 1
            print(
                f"bin size set to {self.bin_size} (belt length {self.belt_length_mm} mm, n_bins {self.n_bins})")
        elif self.bin_size is not None:
            # if belt length / bin size not an integer, round up to cover whole belt with bins. Last belt will be smaller then
            self.n_bins = ceil(self.belt_length_mm / self.bin_size)
            if self.n_bins * self.bin_size != self.belt_length_mm:
                warnings.warn(
                    f"Cannot divide up belt length of {self.belt_length_mm} mm into bins of {self.bin_size} mm! Last bin will have size {self.belt_length_mm % self.bin_size} mm")
            print(
                f"n_bins set to {self.n_bins} (belt length {self.belt_length_mm} mm, bin size {self.bin_size} mm)")
