from omicau.config import OmicauConfig
from omicau.ui.inspect import build_config_dict


def _session(**options):
    return {
        "modalities": [{"name": "rna", "path": "rna.csv"}],
        "clinical": {"path": "clinical.csv", "target": "outcome"},
        **options,
    }


def test_independent_rows_serializes_explicitly_without_model_fitting():
    missing = build_config_dict(_session())
    false = build_config_dict(_session(independent_rows=False))
    true = build_config_dict(_session(independent_rows=True))

    assert missing["cv"]["independent_rows"] is False
    assert false["cv"]["independent_rows"] is False
    assert true["cv"]["independent_rows"] is True
    assert OmicauConfig.from_dict(missing).cv.independent_rows is False
    assert OmicauConfig.from_dict(false).cv.independent_rows is False
    assert OmicauConfig.from_dict(true).cv.independent_rows is True
