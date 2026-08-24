# v1.2 result schema

`analyze_video_structured()` returns an `AnalysisResult` whose serialized form is defined by
[`analysis_result.schema.json`](analysis_result.schema.json). The same schema ships in
OpenSceneSense Ollama so consumers can validate either backend with one contract.

The top-level object contains:

- `schema_version`, fixed to `"1.2"`
- `summary.detailed` and `summary.brief`
- chronological `timeline` events with supporting frame timestamps
- `frame_analyses` and optional `audio_segments`
- video, selection, model, performance, and usage metadata
- explicit `warnings` and `errors`

Provider usage counts are reported but prices are not calculated. Prices change independently of
the library.

Validate a result with Python:

```python
from jsonschema import validate
from openscenesense import analysis_result_schema

validate(result.to_dict(), analysis_result_schema())
```

Use `analyze_video()` for the v1.1 dictionary shape. Its original keys remain and v1.2 metadata is
additive. Regenerate the checked-in schema with `python scripts/export_schema.py`.
