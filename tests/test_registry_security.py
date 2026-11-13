"""joblib/pickle load safety for the ML model registry. Fixtures SYNTHETIC."""
import numpy as np, pytest
from sklearn.linear_model import Ridge
from engine.errors import ArtifactIntegrityError
from engine.memory.store import Store
from engine.models.registry import Registry

@pytest.fixture
def reg(tmp_path):
    r = Registry(Store(tmp_path / "s.db"), tmp_path / "models")
    m = Ridge().fit(np.arange(20.0).reshape(10, 2), np.arange(10.0)); mid, _ = r.register("ridge", "ml_xs", {"a": 1}, "fh", "ds", "cv", 0, model=m)
    return r, mid

def test_roundtrip_loads_verified_artifact(reg):
    r, mid = reg; assert isinstance(r.load_artifact(mid), Ridge)

def test_tampered_file_is_not_loaded(reg):
    r, mid = reg; p = r.dir / r.get(mid)["artifact_path"].split("/")[-1]; p.write_bytes(p.read_bytes() + b"x")
    with pytest.raises(ArtifactIntegrityError, match="hash mismatch"): r.load_artifact(mid)

def test_malicious_pickle_swapped_in_never_executes(reg, tmp_path):
    import pickle
    r, mid = reg; flag = tmp_path / "pwned"
    class Evil:
        def __reduce__(self): return (flag.write_text, ("x",))
    p = r.dir / r.get(mid)["artifact_path"].split("/")[-1]; p.write_bytes(pickle.dumps(Evil()))
    with pytest.raises(ArtifactIntegrityError): r.load_artifact(mid)
    assert not flag.exists()                                           # the payload was never run

def test_db_path_pointing_outside_registry_dir_is_refused(reg, tmp_path):
    r, mid = reg; outside = tmp_path / "outside.joblib"; outside.write_bytes(b"whatever")
    with r.store.tx() as d: d.execute("UPDATE ml_models SET artifact_path=? WHERE id=?", (str(outside), mid))
    with pytest.raises(ArtifactIntegrityError, match="outside"): r.load_artifact(mid)

def test_symlink_escape_is_refused(reg, tmp_path):
    import os
    r, mid = reg; target = tmp_path / "t.joblib"; target.write_bytes(b"x"); link = r.dir / "99999_link.joblib"; os.symlink(target, link)
    with r.store.tx() as d: d.execute("UPDATE ml_models SET artifact_path=? WHERE id=?", (str(link), mid))
    with pytest.raises(ArtifactIntegrityError): r.load_artifact(mid)

def test_legacy_unsealed_row_refused_until_explicit_seal(reg):
    r, mid = reg
    with r.store.tx() as d: d.execute("DELETE FROM ml_artifact_seals WHERE model_id=?", (mid,))
    with pytest.raises(ArtifactIntegrityError, match="no integrity seal"): r.load_artifact(mid)
    r.seal_existing(mid); assert isinstance(r.load_artifact(mid), Ridge)
