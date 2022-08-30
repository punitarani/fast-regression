"""Generate Pandas DataFrame Data"""

import random
import string

import numpy as np
import pandas as pd

from ..errors import InvalidDataType, InvalidModel
from ..types import ModelType


# pylint: disable=too-many-instance-attributes, too-few-public-methods
class GenerateDataFrame:
    """Generate DataFrame"""

    def __init__(
        self,
        n: int = 1,
        m: int = 1,
        model: ModelType | str = ModelType.LINEAR,
        dtype: type | str = float,
        noise: float = 0.0,
    ):
        """
        Constructor

        Args:
            n (int): number of rows to generate
            m (int): number of columns to generate

            model (str): Regression model_type to use

            dtype (str): data type of the DataFrame values
                Allowed values: int, float

        """

        # Input validation
        self.data_type: type = self.__check_dtype(dtype)  # Data type
        self.model_type: ModelType = self.__check_model(model)  # Model type

        # Set attributes
        self.n: int = n  # Number of rows
        self.m: int = m  # Number of columns
        self.model_noise: float = noise  # Model noise

        # Generated data
        self.dataframe: pd.DataFrame | None = None  # Generated DataFrame
        self.cols: list = []  # Column names
        self.model_params: dict = {}  # Model parameters

    def generate(self):
        """
        Main function to generate the DataFrame

        Returns:
            pandas.DataFrame: Generated DataFrame
        """

        # Generate random column names
        self.cols = self.__generate_column_names(self.m)

        # Create DataFrame of the specified Size and dtype
        df = pd.DataFrame(
            index=range(self.n),
            columns=self.cols,
        )

        # Generate data
        for col in self.cols:
            # Generate model parameters
            model = self.__generate_model(self.model_type, self.n)
            self.model_params[col] = model

            # Generate column data
            df[col] = self.__generate_values(
                self.data_type, self.n, model, self.model_noise
            )

        # Enforce dtype
        df = df.astype(self.data_type)

        # Cache dataframe
        self.dataframe = df

        return df

    # region Static Methods

    @staticmethod
    def __check_dtype(dtype: type) -> type:
        """
        Check if data type is valid

        Args:
            dtype (str): data type of the DataFrame values
                Allowed values: int, float

        Raises:
            InvalidDataType Exception if dtype is invalid

        Returns:
            None
        """
        # type input validation
        if isinstance(dtype, type):
            # Check if data type is valid
            if dtype not in [int, float]:
                raise InvalidDataType(dtype)

            # Return if valid
            return dtype

        # str input validation
        else:
            # Check if dtype is a valid data type
            if dtype not in ["str", "float"]:
                raise InvalidDataType(dtype)

            # Convert to type
            if dtype == "int":
                return int

            return float

    @staticmethod
    def __check_model(model: ModelType | str) -> ModelType:
        """
        Get model_type type

        Args:
            model (ModelType | str): Model Type Enum or string

        Returns:
            ModelType: Model Type Enum
        """

        # Return model_type if it is a ModelType Enum
        if isinstance(model, ModelType):
            return model

        # Match model_type string to ModelType Enum
        else:
            if model.lower() == "linear":
                return ModelType.LINEAR

            raise InvalidModel(model)

    @staticmethod
    def __generate_column_names(n: int):
        """
        Generate column names

        Args:
            n (int): number of columns to generate

        Returns:
            list: Random column names (lowercase) of length 5.
        """

        # Generate random string of length 5
        return ["".join(random.choices(string.ascii_lowercase, k=5)) for _ in range(n)]

    @staticmethod
    def __generate_model(model: ModelType, n: int = 100) -> list:
        """
        Generate model parameters

        Args:
            model (ModelType): Model Type Enum

        Returns:
            list: Model parameters [b0, b1, ...]
        """

        # Linear model
        if model == ModelType.LINEAR:
            return [
                round(random.uniform(-10*n, 10*n)),
                round(random.uniform(-n/100, n/100)),
            ]

        return []

    @staticmethod
    def __generate_values(
        dtype: type,
        n: int,
        model: list,
        noise: float,
    ) -> np.ndarray:
        """
        Generates Values for the DataFrame Column

        Args:
            dtype (str): data type of the DataFrame values
                Allowed values: int, float
            n (int): number of rows
            model (str): Model parameters

        Returns:
            np.ndarray: Generated values
        """

        # Generate index
        index = np.arange(n)

        # Generate data
        data = np.zeros(n)

        for pwr, val in enumerate(model):
            data += val * np.power(index, pwr)

        # Add noise
        if noise > 0:
            data += np.random.normal(0, noise * n, n)

        return data

    # endregion Static Methods
