import numpy as np

from qkef.evolution.features import CONVENTIONAL_FEATURE_NAMES, QUANTUM_FEATURE_NAMES, assert_feature_manifest_safe, conventional_features, quantum_features
from qkef.evolution.models import CLASS_ORDER, save_reload_verify, select_model


def test_features_are_finite_and_fixed_width():
    conv = conventional_features("alpha 2026", ["alpha 2025"], [0.8])
    quant = quantum_features(np.array([1.0, 0.0]), [np.array([0.8, 0.6])])
    assert conv.shape == (len(CONVENTIONAL_FEATURE_NAMES),)
    assert quant.shape == (len(QUANTUM_FEATURE_NAMES),)
    assert np.isfinite(np.r_[conv, quant]).all()


def test_feature_manifest_rejects_leaky_name():
    try:
        assert_feature_manifest_safe(["target_action"])
    except ValueError:
        pass
    else:
        raise AssertionError("leaky feature should fail")


def test_model_selection_and_reload(tmp_path):
    x = np.vstack([np.arange(6) + offset for offset in range(12)]).astype(float)
    y = np.array(list(CLASS_ORDER) * 2)
    model, result = select_model(x, y, x, y, [0.1, 1.0])
    assert result["C"] in {0.1, 1.0}
    assert save_reload_verify(model, tmp_path / "model.joblib", x)
