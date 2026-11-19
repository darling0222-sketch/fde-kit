from __future__ import annotations

import numpy as np

from fde_kit.models import Asset, Chunk
from fde_kit.store import Store


def _asset(asset_id: str = "as_1", sha: str = "abc") -> Asset:
    return Asset(
        id=asset_id,
        source="local",
        uri=f"examples/{asset_id}.md",
        media_type="md",
        title="测试文档",
        sha256=sha,
        size=12,
        text="正文",
    )


def test_asset_roundtrip_and_dedupe(tmp_path):
    store = Store(tmp_path / "t.db")
    store.upsert_asset(_asset())
    assert store.count_assets() == 1
    assert store.find_asset_by_sha("abc").id == "as_1"
    store.upsert_asset(_asset())  # 幂等
    assert store.count_assets() == 1
    store.close()


def test_chunks_replace_and_cascade(tmp_path):
    store = Store(tmp_path / "t.db")
    store.upsert_asset(_asset())
    store.add_chunks(
        [
            Chunk(id="ck_1", asset_id="as_1", ord=0, text="第一段", tokens=3),
            Chunk(id="ck_2", asset_id="as_1", ord=1, text="第二段", tokens=3),
        ]
    )
    assert store.count_chunks() == 2
    store.replace_asset_chunks("as_1", [Chunk(id="ck_9", asset_id="as_1", ord=0, text="新的", tokens=2)])
    assert store.count_chunks() == 1
    store.delete_asset("as_1")
    assert store.count_chunks() == 0 and store.count_assets() == 0
    store.close()


def test_vector_roundtrip(tmp_path):
    store = Store(tmp_path / "t.db")
    vec = np.asarray([[0.6, 0.8, 0.0]], dtype=np.float32)
    store.set_vectors({"ck_1": vec[0]})
    ids, mat = store.load_vectors()
    assert ids == ["ck_1"]
    assert mat.shape == (1, 3)
    assert np.allclose(mat[0], vec[0])
    store.close()


def test_revision_and_kv(tmp_path):
    store = Store(tmp_path / "t.db")
    assert store.revision() == 0
    store.bump_revision()
    assert store.revision() == 1
    store.kv_set("x", "y")
    assert store.kv_get("x") == "y"
    store.close()


def test_runs_and_evals(tmp_path):
    store = Store(tmp_path / "t.db")
    run_id = store.create_run("eval", {"cases": 2})
    store.save_eval(run_id, "c1", {"hit": True})
    store.finish_run(run_id, "done", {"summary": {"hit_rate": 1.0}})
    run = store.get_run(run_id)
    assert run["status"] == "done"
    assert run["meta"]["cases"] == 2
    assert run["meta"]["summary"]["hit_rate"] == 1.0
    assert store.list_evals(run_id)[0]["case_id"] == "c1"
    store.close()
