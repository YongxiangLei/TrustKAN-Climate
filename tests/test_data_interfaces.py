import numpy as np
import pandas as pd
import pytest

from src.data.common import from_dataframe
from src.data.ghcn import (
    ghcn_station_url,
    prepare_ghcn_element,
    prepare_ghcn_temperature_pair,
    read_ghcn_archive,
    read_ghcn_inventory,
)
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


def test_ghcn_fixed_width_inventory_parser(tmp_path):
    path=tmp_path/"inventory.txt"
    path.write_text("ACW00011604  17.1167  -61.7833 TMAX 1949 2025\n",encoding="ascii")
    parsed=read_ghcn_inventory(path)
    assert parsed.loc[0,"ID"]=="ACW00011604"
    assert parsed.loc[0,"ELEMENT"]=="TMAX"
    assert parsed.loc[0,"FIRST_YEAR"]==1949
    assert parsed.loc[0,"LAST_YEAR"]==2025


def test_ghcn_temperature_pair_uses_quality_controlled_date_intersection():
    station="USW00094728"
    raw=pd.DataFrame(
        [
            [station,"20200101","TMAX","100",pd.NA,pd.NA,"Z",pd.NA],
            [station,"20200101","TMIN","0",pd.NA,pd.NA,"Z",pd.NA],
            [station,"20200102","TMAX","120",pd.NA,pd.NA,"Z",pd.NA],
            [station,"20200102","TMIN","20",pd.NA,"X","Z",pd.NA],
        ],
        columns=["ID","DATE","ELEMENT","DATA_VALUE","M_FLAG","Q_FLAG","S_FLAG","OBS_TIME"],
    )
    paired=prepare_ghcn_temperature_pair(raw,station)
    assert paired.date.tolist()==[pd.Timestamp("2020-01-01")]
    assert paired.loc[0,"tmax"]==10.0
    assert paired.loc[0,"tmin"]==0.0
    assert paired.loc[0,"target"]==5.0


def test_ghcn_temperature_pair_rejects_inverted_extrema():
    station="TEST0000001"
    raw=pd.DataFrame(
        [
            [station,"20200101","TMAX","100",pd.NA,pd.NA,"Z",pd.NA],
            [station,"20200101","TMIN","120",pd.NA,pd.NA,"Z",pd.NA],
        ],
        columns=["ID","DATE","ELEMENT","DATA_VALUE","M_FLAG","Q_FLAG","S_FLAG","OBS_TIME"],
    )
    paired=prepare_ghcn_temperature_pair(raw,station)
    assert paired.empty


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
