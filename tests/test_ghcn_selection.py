from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

from src.data.ghcn import select_temperature_candidates, verify_frozen_station_panel


def inventory_row(station, element, first, last, latitude=40.0, longitude=-80.0):
    return {
        "ID": station,
        "LATITUDE": latitude,
        "LONGITUDE": longitude,
        "ELEMENT": element,
        "FIRST_YEAR": first,
        "LAST_YEAR": last,
    }


def test_candidate_selection_requires_both_temperature_elements_and_is_deterministic():
    rows=[
        inventory_row("STA00000002","TMAX",1940,2025),
        inventory_row("STA00000002","TMIN",1945,2025),
        inventory_row("STA00000001","TMAX",1940,2025),
        inventory_row("STA00000001","TMIN",1945,2025),
        inventory_row("STA00000003","TMAX",1930,2025),
    ]
    regions=[
        {
            "name":"region",
            "latitude_min":30,
            "latitude_max":50,
            "longitude_min":-100,
            "longitude_max":-60,
        }
    ]
    selected=select_temperature_candidates(
        pd.DataFrame(rows),regions,required_start_year=1950,required_end_year=2024,candidates_per_region=2
    )
    assert selected.ID.tolist()==["STA00000001","STA00000002"]
    assert selected.CANDIDATE_RANK.tolist()==[1,2]
    assert "STA00000003" not in selected.ID.tolist()


def test_candidate_selection_respects_fixed_region_bounds():
    rows=[
        inventory_row("IN000000001","TMAX",1940,2025,latitude=10,longitude=10),
        inventory_row("IN000000001","TMIN",1940,2025,latitude=10,longitude=10),
    ]
    regions=[
        {
            "name":"north",
            "latitude_min":30,
            "latitude_max":60,
            "longitude_min":-20,
            "longitude_max":40,
        }
    ]
    selected=select_temperature_candidates(
        pd.DataFrame(rows),regions,required_start_year=1950,required_end_year=2024
    )
    assert selected.empty


def test_frozen_panel_has_one_unique_station_per_registered_region():
    config_path=Path(__file__).parents[1]/"configs"/"datasets"/"ghcn_frozen.yaml"
    with open(config_path,"r",encoding="utf-8") as handle:
        config=yaml.safe_load(handle)
    stations=config["stations"]
    assert len(stations)==5
    assert len({item["region"] for item in stations})==5
    assert len({item["station_id"] for item in stations})==5
    assert all(len(item["raw_sha256"])==64 for item in stations)
    assert config["dataset"]["selection"]["model_performance_used"] is False


def test_frozen_panel_verification_rejects_station_drift():
    frozen={
        "dataset":{"inventory_sha256":"inventory"},
        "stations":[
            {"region":"region","station_id":"EXPECTED","candidate_rank":1,"raw_sha256":"raw"}
        ],
    }
    selected=[
        {"region":"region","station_id":"CHANGED","candidate_rank":1,"raw_sha256":"raw"}
    ]
    with pytest.raises(ValueError,match="station_id mismatch"):
        verify_frozen_station_panel(frozen,"inventory",selected)
