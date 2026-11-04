- **2026-11-04** -- Engine patch `0098`: a tool parameter whose schema names no type (`{}`, a description alone, a `$ref`
  that resolves nowhere) takes the JSON object or array the model wrote instead of coming back as a JSON string of it;
  scalars, plain words and quoted strings stay text, and a tool declared without a parameter schema keeps streaming its
  arguments as before. The rule is the one agreed on the 2x DGX Sparks recipe's issue #180 (upstream keeps untyped
  values as text; noted at the gate). Ported from the GB10 pair's recipe
  (MiaAI-Lab/GLM-5.3-Flash-DGX-Sparks-1104, its `0105`), where the bug was reproduced live and the patch host-tested;
  on this stack `zig build test-tool-parse-host test-tool-visibility-http-host test-json-answer-host
  check-native-cuda-host` pass (79 tests). Takes effect at the next `./start.sh restart` (new image).
