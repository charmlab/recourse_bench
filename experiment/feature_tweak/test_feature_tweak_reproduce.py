from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiment.utils import write_reproduction_report  # noqa: E402
from method.feature_tweak.model import FeatureTweak  # noqa: E402
from method.feature_tweak.support import (  # noqa: E402
    FeatureTweakContext,
    FeatureTweakGroup,
)


REPORT_PATH = Path(__file__).with_name("reproduction_report.json")
FEATURE_NAMES = ["static_score", "mutable_quality", "static_noise"]


class ForestAdapter:
    def __init__(self, forest: RandomForestClassifier) -> None:
        self._forest = forest
        self.feature_input_order = list(FEATURE_NAMES)
        self.tree_iterator = tuple(forest.estimators_)
        self.classes_ = np.asarray([0, 1], dtype=np.int64)

    def get_ordered_features(self, values: pd.DataFrame | np.ndarray) -> pd.DataFrame:
        if isinstance(values, pd.DataFrame):
            return values.loc[:, self.feature_input_order].copy(deep=True)
        array = np.asarray(values, dtype="float64")
        if array.ndim == 1:
            array = array.reshape(1, -1)
        return pd.DataFrame(array, columns=self.feature_input_order)

    def predict(self, values: pd.DataFrame | np.ndarray) -> np.ndarray:
        return self._forest.predict(self.get_ordered_features(values))


def _feature_context() -> FeatureTweakContext:
    return FeatureTweakContext(
        feature_names=tuple(FEATURE_NAMES),
        target_column="target",
        groups=(
            FeatureTweakGroup(
                "static_score",
                ("static_score",),
                (0,),
                "scalar",
                "numerical",
                False,
                "none",
                None,
                0.0,
                1.0,
                False,
            ),
            FeatureTweakGroup(
                "mutable_quality",
                ("mutable_quality",),
                (1,),
                "scalar",
                "numerical",
                True,
                "any",
                None,
                0.0,
                1.0,
                False,
            ),
            FeatureTweakGroup(
                "static_noise",
                ("static_noise",),
                (2,),
                "scalar",
                "numerical",
                False,
                "none",
                None,
                0.0,
                1.0,
                False,
            ),
        ),
    )


def _training_data() -> tuple[pd.DataFrame, np.ndarray]:
    rows: list[list[float]] = []
    labels: list[int] = []
    for static_score in [0.0, 1.0]:
        for mutable_quality in np.linspace(0.0, 1.0, 11):
            for static_noise in [0.0, 1.0]:
                rows.append([static_score, float(mutable_quality), static_noise])
                labels.append(int(mutable_quality > 0.55))
    return pd.DataFrame(rows, columns=FEATURE_NAMES), np.asarray(labels, dtype=np.int64)


def _factuals() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "static_score": [0.0, 1.0, 1.0],
            "mutable_quality": [0.1, 0.2, 0.3],
            "static_noise": [1.0, 0.0, 1.0],
        }
    )


@pytest.mark.fast
def test_reproduce() -> None:
    """
    Scoped reproduction of actionable feature tweaking.

    The paper evaluates Yahoo Gemini advertisements, which are not available in
    this checkout. This toy forest preserves the central claim: recommend
    outcome-changing tree-path edits using only features marked actionable.
    """
    x_train, y_train = _training_data()
    forest = RandomForestClassifier(
        n_estimators=5,
        max_depth=2,
        random_state=1,
    ).fit(x_train, y_train)
    method = FeatureTweak(
        mlmodel=ForestAdapter(forest),
        context=_feature_context(),
        desired_class=1,
        eps=0.02,
    )

    factuals = _factuals()
    counterfactuals = method.get_counterfactuals(factuals)
    predictions = forest.predict(counterfactuals)
    static_columns = ["static_score", "static_noise"]
    static_violations = int(
        np.sum(
            ~np.isclose(
                counterfactuals.loc[:, static_columns].to_numpy(dtype="float64"),
                factuals.loc[:, static_columns].to_numpy(dtype="float64"),
            )
        )
    )
    mutable_changes = np.abs(
        counterfactuals["mutable_quality"].to_numpy(dtype="float64")
        - factuals["mutable_quality"].to_numpy(dtype="float64")
    )

    assert float(np.mean(predictions == 1)) == pytest.approx(1.0)
    assert static_violations == 0
    assert float(np.mean(mutable_changes > 0.0)) == pytest.approx(1.0)

    write_reproduction_report(
        REPORT_PATH,
        paper_id="Interpretable Predictions of Tree-based Ensembles via Actionable Feature Tweaking",
        reproduction_metadata={
            "source_script": Path(__file__).name,
            "paper_reference": "reference/feature_tweak.pdf",
            "paper_anchor": "Yahoo Gemini actionable feature tweaking setup and offline evaluation",
            "scope": "Scoped synthetic tree-ensemble check; Yahoo Gemini data are unavailable.",
            "timestamp_utc": datetime.now(timezone.utc),
        },
        experiments_data={
            "actionable_tree_path_tweaks": {
                "configuration": {
                    "num_factuals": len(factuals),
                    "actionable_features": ["mutable_quality"],
                    "static_features": static_columns,
                    "forest_n_estimators": 5,
                    "forest_max_depth": 2,
                },
                "metrics": {
                    "target_flip_rate": {
                        "original": None,
                        "reproduced": float(np.mean(predictions == 1)),
                    },
                    "static_feature_violations": {
                        "original": 0,
                        "reproduced": static_violations,
                    },
                    "mean_mutable_feature_change": {
                        "original": None,
                        "reproduced": float(mutable_changes.mean()),
                    },
                    "mutable_feature_change_rate": {
                        "original": None,
                        "reproduced": float(np.mean(mutable_changes > 0.0)),
                    },
                },
            }
        },
    )


if __name__ == "__main__":
    test_reproduce()
