import numpy as np
import pandas as pd
import pytest

from src.data.common import from_dataframe
from src.data.ghcn import ghcn_station_url, prepare_ghcn_element, read_ghcn_archive
from src.data.era5 import era5_timeseries_request


def test_ghcn_url_validation():
    assert ghcn_station_url("USW00094728").endswith("USW00094728.csv.gz")
    with pytest.raises(ValueError):
        ghcn_station_url("../bad")


def test_ghcn_headerless_archive_and_quality_control(tmp_path):
    station="USW00094728"
    rows=[
        [station,"20200101","TAVG","100","","","Z","1200"],
        [station,"20200102","TAVG","110","","X","Z","1200"],
        [station,"20200103","TAVG","-9999","","","Z","1200"],
        [station,"20200104","TMAX","200","","","Z","1200"],
    ]
    path=tmp_path/"station.csv.gz"
    pd.DataFrame(rows).to_csv(path,index=False,header=False,compression="gzip")
    raw=read_ghcn_archive(path)
    assert len(raw)==4
    prepared=prepare_ghcn_element(raw,station,"TAVG")
    assert len(prepared)==1
    assert prepared.loc[0,"date"]==pd.Timestamp("2020-01-01")
    assert prepared.loc[0,"value"]==10.0


def test_ghcn_period_filter_is_inclusive():
    station="USW00094728"
    raw=pd.DataFrame(
        [
            [station,"20200101","TAVG","100",pd.NA,pd.NA,"Z",pd.NA],
            [station,"20200102","TAVG","110",pd.NA,pd.NA,"Z",pd.NA],
            [station,"20200103","TAVG","120",pd.NA,pd.NA,"Z",pd.NA],
        ],
        columns=["ID","DATE","ELEMENT","DATA_VALUE","M_FLAG","Q_FLAG","S_FLAG","OBS_TIME"],
    )
    prepared=prepare_ghcn_element(raw,station,start="2020-01-02",end="2020-01-03")
    assert prepared.date.tolist()==[pd.Timestamp("2020-01-02"),pd.Timestamp("2020-01-03")]


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
