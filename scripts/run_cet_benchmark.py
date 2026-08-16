"""Leakage-safe CET benchmark with neural and classical model registry."""
from __future__ import annotations
import argparse, json, platform, time, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
import yaml
import _bootstrap  # noqa: F401  # repository-root import setup
from src.data.timeseries import chronological_split, sliding_windows, TrainOnlyStandardizer, assign_windows_by_target_origin
from src.metrics.forecast import mae, rmse
from src.models.baselines import MLPForecaster, RNNForecaster, TransformerForecaster, PersistenceForecaster
from src.models.kan_baseline import StandardKANForecaster
from src.models.trustkan import TrustKAN
from src.models.advanced_baselines import TCNForecaster, MambaForecaster
from src.models.tem2kan import Tem2KANReference
from src.models.classical import make_svr, make_random_forest, make_xgboost
from src.training.engine import set_seed, train_regressor, predict

NEURAL = {"mlp", "lstm", "gru", "transformer", "tcn", "mamba", "kan", "tem2kan", "trustkan"}
CLASSICAL = {"svr", "random_forest", "xgboost"}

def load_config(path):
    with open(path, "r", encoding="utf-8") as f: return yaml.safe_load(f)

def ensure_data(cfg, cache_dir=Path("data/raw")):
    cache_dir.mkdir(parents=True, exist_ok=True); dest=cache_dir/"cet_mean_station_series.csv"
    if not dest.exists(): urllib.request.urlretrieve(cfg["dataset"]["source"], dest)
    return dest

def build_model(name, history, horizon):
    if name=="mlp": return MLPForecaster(history,1,horizon)
    if name=="lstm": return RNNForecaster("lstm",1,horizon)
    if name=="gru": return RNNForecaster("gru",1,horizon)
    if name=="transformer": return TransformerForecaster(1,horizon)
    if name=="tcn": return TCNForecaster(1,horizon)
    if name=="mamba": return MambaForecaster(1,horizon)
    if name=="kan": return StandardKANForecaster(history,1,horizon)
    if name=="tem2kan": return Tem2KANReference(history,1,horizon)
    if name=="trustkan": return TrustKAN(1,horizon=horizon,hidden_dim=64,grid_size=8)
    if name=="svr": return make_svr()
    if name=="random_forest": return make_random_forest()
    if name=="xgboost": return make_xgboost()
    raise KeyError(f"Unknown model: {name}")

def loader(x,y,batch,shuffle=False):
    return DataLoader(TensorDataset(torch.tensor(x,dtype=torch.float32),torch.tensor(y,dtype=torch.float32)),batch_size=batch,shuffle=shuffle)

def inverse_target(scaler,a):
    a=np.asarray(a); shape=a.shape
    return scaler.scaler.inverse_transform(a.reshape(-1,1)).reshape(shape)

def environment():
    return {"python":platform.python_version(),"torch":torch.__version__,"device":"cuda" if torch.cuda.is_available() else "cpu"}

def main(config):
    cfg=load_config(config); path=ensure_data(cfg)
    d=pd.read_csv(path,parse_dates=[cfg["dataset"]["date_column"]]); s=d[cfg["dataset"]["target_column"]].dropna()
    values=s[s>=cfg["dataset"]["min_valid_temperature"]].astype(float).values
    max_observations=cfg["dataset"].get("max_observations")
    if max_observations is not None:
        if not isinstance(max_observations,int) or max_observations<=0:
            raise ValueError("dataset.max_observations must be a positive integer")
        values=values[-max_observations:]
    split=chronological_split(len(values),cfg["split"]["train"],cfg["split"]["validation"],cfg["split"]["calibration"])
    scaler=TrainOnlyStandardizer().fit(values[split.train]); z=scaler.transform(values)
    rows=[]; outdir=Path("results/raw"); outdir.mkdir(parents=True,exist_ok=True)
    for horizon in cfg["window"]["horizons"]:
        X,y,origins=sliding_windows(z,cfg["window"]["history"],horizon); masks=assign_windows_by_target_origin(origins,split,horizon)
        sets={k:(X[m],y[m]) for k,m in masks.items()}
        test_x,test_y=sets["test"]; test_y_raw=inverse_target(scaler,test_y)
        for name in cfg["models"]:
            seeds=[-1] if name=="persistence" else cfg["training"]["seeds"]
            for seed in seeds:
                row={"dataset":"CET-Pershore","model":name,"horizon":horizon,"seed":seed,"status":"ok",**environment()}
                try:
                    start=time.perf_counter()
                    if name=="persistence":
                        pred_std=PersistenceForecaster(horizon).predict(test_x).numpy(); seconds=0.0; params=0
                    elif name in CLASSICAL:
                        set_seed(seed); model=build_model(name,cfg["window"]["history"],horizon)
                        model.fit(*sets["train"]); pred_std=model.predict(test_x); seconds=time.perf_counter()-start; params=np.nan
                    elif name in NEURAL:
                        set_seed(seed); model=build_model(name,cfg["window"]["history"],horizon)
                        tr=loader(*sets["train"],cfg["training"]["batch_size"],True); va=loader(*sets["val"],cfg["training"]["batch_size"]); te=loader(*sets["test"],cfg["training"]["batch_size"])
                        model,_,seconds=train_regressor(model,tr,va,epochs=cfg["training"]["epochs"],lr=cfg["training"]["learning_rate"],patience=cfg["training"]["patience"])
                        pred_std,_=predict(model,te); params=sum(p.numel() for p in model.parameters())
                    else: raise KeyError(name)
                    pred=inverse_target(scaler,pred_std)
                    np.savez_compressed(outdir/f"cet_{name}_h{horizon}_s{seed}.npz",prediction=pred,target=test_y_raw)
                    row.update(rmse=rmse(test_y_raw,pred),mae=mae(test_y_raw,pred),train_seconds=seconds,parameters=params)
                except Exception as exc:
                    row.update(status="failed",error=f"{type(exc).__name__}: {exc}")
                    print(f"FAILED {name} h={horizon} seed={seed}: {exc}")
                rows.append(row)
    result=pd.DataFrame(rows); Path("results/aggregated").mkdir(parents=True,exist_ok=True)
    result.to_csv("results/aggregated/cet_runs.csv",index=False)
    ok=result[result.status=="ok"]
    summary=ok.groupby(["model","horizon"],as_index=False).agg(rmse_mean=("rmse","mean"),rmse_sd=("rmse","std"),mae_mean=("mae","mean"),mae_sd=("mae","std"),train_seconds_mean=("train_seconds","mean"))
    summary.to_csv("results/aggregated/cet_summary.csv",index=False); print(summary.to_string(index=False))
    with open("results/aggregated/environment.json","w") as f: json.dump(environment(),f,indent=2)

if __name__=="__main__":
    ap=argparse.ArgumentParser(); ap.add_argument("--config",default="configs/cet.yaml"); main(ap.parse_args().config)
