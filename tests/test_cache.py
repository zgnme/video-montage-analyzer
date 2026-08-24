import os

import pytest

from openscenesense.cache import AnalysisCache, analysis_cache_key


def test_cache_key_is_stable_and_configuration_sensitive(tmp_path):
    video = tmp_path / "video.bin"
    video.write_bytes(b"a" * 32)
    first = analysis_cache_key(str(video), {"model": "one"})
    second = analysis_cache_key(str(video), {"model": "one"})
    changed = analysis_cache_key(str(video), {"model": "two"})

    assert first == second
    assert changed != first


def test_cache_hashes_both_edges_of_large_files(tmp_path):
    video = tmp_path / "video.bin"
    video.write_bytes(b"a" * 32)
    first = analysis_cache_key(str(video), {})
    stat = video.stat()
    video.write_bytes(b"a" * 31 + b"b")
    os.utime(video, ns=(stat.st_atime_ns, stat.st_mtime_ns))

    assert analysis_cache_key(str(video), {}) != first


def test_cache_recovers_from_corrupt_json_and_blocks_escape(tmp_path):
    cache = AnalysisCache(tmp_path, "key")
    target = cache.path("result.json")
    target.parent.mkdir(parents=True)
    target.write_text("{not-json", encoding="utf-8")

    assert cache.read_json("result.json") is None
    cache.write_json("frame_analyses/0000.json", {"ok": True})
    assert cache.read_json("frame_analyses/0000.json") == {"ok": True}
    with pytest.raises(ValueError, match="escapes"):
        cache.path("../outside.json")
