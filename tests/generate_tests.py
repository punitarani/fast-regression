"""Test generate_*() functions"""

import pytest
from models import InvalidDataType, InvalidModel, ModelType
from models.generate import GenerateDataFrame


class TestGenerateDataframe:
    """Test GenerateDataFrame"""

    def test_init(self):
        """Test __init()"""

        gendf = GenerateDataFrame()

        assert gendf.n == 1
        assert gendf.m == 1
        assert gendf.model_type == ModelType.LINEAR
        assert gendf.data_type == float
        assert gendf.model_noise == 0.0

        assert gendf.dataframe is None
        assert not gendf.cols
        assert not gendf.model_params

    def test_generate(self):
        """Test generate()"""

        gendf = GenerateDataFrame(n=100, m=10, model="linear", dtype=int, noise=0.1)

        assert gendf.n == 100
        assert gendf.m == 10
        assert gendf.model_type == ModelType.LINEAR
        assert gendf.data_type == int
        assert gendf.model_noise == 0.1

        gendf.generate()

        assert gendf.dataframe is not None
        assert len(gendf.cols) == 10
        assert len(gendf.model_params) == 10

    def test_invalid_dtype(self):
        """Test invalid dtype checking"""

        with pytest.raises(InvalidDataType):
            GenerateDataFrame(dtype=str)

    def test_invalid_model(self):
        """Test invalid model checking"""

        with pytest.raises(InvalidModel):
            GenerateDataFrame(model="invalid")
