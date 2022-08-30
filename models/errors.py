"""Errors"""


class InvalidDataType(Exception):
    """Invalid Data Type Exception"""

    def __init__(self, dtype: type):
        """
        InvalidDataType exception Constructor

        Args:
            dtype (type): Invalid data type
        """

        self.dtype = dtype

        super().__init__(f"Invalid data type: {self.dtype}")


class InvalidModel(Exception):
    """Invalid Model Exception"""

    def __init__(self, model: str):
        """
        InvalidModel exception Constructor

        Args:
            model (str): Invalid regression model_type
        """

        self.model = model

        super().__init__(f"Invalid model_type: {self.model}")
