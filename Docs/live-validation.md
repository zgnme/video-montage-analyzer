# Live release validation

These opt-in checks were run on 2026-08-24 in addition to the mocked provider test suite. API keys
were loaded from environment variables and were never written to results or logs.

## Real audiovisual fixtures

- The full 596.46-second Blender Foundation *Big Buck Bunny* MP4 was downloaded from
  `https://download.blender.org/peach/bigbuckbunny_movies/BigBuckBunny_320x180.mp4.zip`.
- The extracted 64,657,027-byte MP4 had SHA-256
  `f78f39603e6774907f2faafabf26a6674a6fc31769ec304a8a8f7c62d280508`, H.264 video, and stereo
  AAC audio.
- The bundled 30.69-second `Examples/genvideo.mp4` supplied narrated speech coverage.

The long fixture is intentionally not committed or included in release artifacts.

## OpenAI

`gpt-5.6-luna` analyzed ten dynamically selected frames from the full film and generated the final
summary in 31.02 seconds. The result contained ten frame analyses and ten events, validated against
the v1.2 schema, and contained no warnings or errors. Reported usage was 6,272 tokens.

The narrated fixture was also processed with four frames plus `whisper-1`. It produced five
timestamped audio segments and six events in 19.93 seconds with no warnings or errors.

## OpenRouter

`stealth/ox-alpha` completed the generated two-scene integration with two frame analyses and a
validated summary. The live model ignored `response_format` for an image request, which led to the
explicit schema instruction and single validated text-repair path shipped in v1.2.

A ten-frame full-film attempt was interrupted by OpenRouter's shared upstream model pool returning
HTTP 429 responses. Three completed frames were preserved through the resumable cache. This is kept
distinct from schema or pipeline validation: the smaller live integration passed, while the
external shared-pool limit prevented a completed long run.
