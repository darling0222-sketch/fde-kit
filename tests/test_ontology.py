from __future__ import annotations

from pathlib import Path

from fde_kit.compile.ontology import Ontology

ROOT = Path(__file__).resolve().parents[1]


def test_extract_finds_structured_entities(ontology: Ontology):
    text = "A100 报 E021，备件号 P-2107，设备序列号 SN-2024-0311，工单 WO-2509-0112。"
    found = {(etype, value) for etype, value, _s, _e in ontology.extract(text)}
    assert ("Product", "A100") in found
    assert ("FaultCode", "E021") in found
    assert ("PartNo", "P-2107") in found
    assert ("SerialNo", "SN-2024-0311") in found
    assert ("WorkOrder", "WO-2509-0112") in found


def test_extract_deduplicates_same_value(ontology: Ontology):
    text = "A100 过热。A100 再次过热。A100 反复过热。"
    products = [v for t, v, _s, _e in ontology.extract(text) if t == "Product"]
    assert products == ["A100"]


def test_match_query_uses_alias_with_lower_weight(ontology: Ontology):
    hits = ontology.match_query("这台整机的保修期怎么算")
    types = {t for t, _v, _w in hits}
    assert "Product" in types
    assert "PolicyTerm" in types
    alias_hit = [w for t, v, w in hits if v == "整机"]
    assert alias_hit and alias_hit[0] < 0.2


def test_coverage_reports_missing_types(ontology: Ontology):
    rows = ontology.coverage({"Product": 3})
    by_type = {r["type"]: r for r in rows}
    assert by_type["Product"]["covered"] is True
    assert by_type["FaultCode"]["covered"] is False


def test_missing_file_falls_back_to_empty(tmp_path):
    empty = Ontology.load(tmp_path / "nope.yaml")
    assert empty.entity_types == {}
    assert empty.extract("A100 E021") == []
