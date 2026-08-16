import numpy as np
import pandas as pd
import pytest

from src.data.common import from_dataframe
from src.data.ghcn import ghcn_station_url
from src.data.era5 import era5_timeseries_request


def test_ghcn_url_validation():
    assert ghcn_station_url("USW00094728").endswith("USW00094728.csv.gz")
    with pytest.raises(ValueError):
        ghcn_station_url("../bad")


def test_common_contract_sorts_and_validates():
    df=pd.DataFrame({"time":["2020-01-02","2020-01-01"],"x":[2.,1.],"y":[4.,3.]})
    s=from_dataframe("toy",df,"time",["x"],"y")
    assert s.dates.is_monotonic_increasing
    assert s.features.shape==(2,1)
    assert np.allclose(s.target,[3.,4.])


def test_era5_request_shape():
    r=era5_timeseries_request("2m_temperature",2020,1,50.91,11.56)
    assert r["year"]==["2020"]
    assert len(r["time"])==24
    assert r["location"]["latitude"]==50.91
