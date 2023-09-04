import logging
from typing import Union
import pandas as pd
from lida.utils import read_dataframe

logger = logging.getLogger(__name__)

class Preprocessor():
    def __init__(self) -> None:
        self.cleaned_data = None

    def fill_na(self, data: pd.DataFrame) -> pd.DataFrame:
        """Fill missing values in a pandas DataFrame"""
        filled_na = data.fillna("")
        return filled_na

    def preprocess(
        self, data: Union[pd.DataFrame, str], file_name: str = "", n_samples: int = 3,
    ):
        """Preprocesses data from a pandas DataFrame or a file location"""

        # if data is a file path, read it into a pandas DataFrame, set file_name to the file name
        if isinstance(data, str):
            file_name = data.split("/")[-1]
            data = read_dataframe(data)
        filled_na = self.fill_na(data)
        cleaned_data = filled_na
        self.cleaned_data = cleaned_data
        return cleaned_data