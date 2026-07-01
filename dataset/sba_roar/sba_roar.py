from __future__ import annotations

from pathlib import Path

import pandas as pd

from dataset.dataset_object import DatasetObject
from utils.registry import register


@register("sba_roar")
class SbaRoarDataset(DatasetObject):
    def __init__(self, path: str = "./dataset/sba_roar/", **kwargs):
        dataset_path = Path(path)
        if not dataset_path.exists():
            dataset_path = Path(__file__).resolve().parent

        self._rawdf = self._read_df(str(dataset_path))
        self._freeze = False

        rawattrs = self._read_attrs(str(dataset_path))
        for flag, value in rawattrs.items():
            setattr(self, flag, value)

        feature_order = getattr(self, "feature_order", list(self._rawdf.columns))
        self._rawdf = self._rawdf.loc[:, feature_order].copy(deep=True)

    def _read_df(self, path: str) -> pd.DataFrame:
        df_path = Path(path) / "SBAcase.11.13.17.csv"
        df = pd.read_csv(df_path)
        df.columns = [column.lstrip("\ufeff") for column in df.columns]
        df = df.fillna(-1)
        df["NoDefault"] = 1 - df["Default"].astype(int)
        return df.drop(
            columns=[
                "Selected",
                "State",
                "Name",
                "BalanceGross",
                "LowDoc",
                "BankState",
                "LoanNr_ChkDgt",
                "MIS_Status",
                "Default",
                "Bank",
                "City",
            ]
        )
