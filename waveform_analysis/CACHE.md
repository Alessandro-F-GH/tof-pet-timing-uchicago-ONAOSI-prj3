# Named cache layout

The waveform pipeline writes reusable data directly to human-readable directories. A cache root is selected through `preprocessing.cache_dir`; when using `--cache-root BUNDLE`, the cache root is `BUNDLE/preprocessing`.

## Layout

```text
preprocessing/
  control/<control-dataset>/<modes>/
  selection/<control-dataset>/<dataset>/<energy-or-timing>/
  native/<control-dataset>/<dataset>/<energy-or-timing>/
  datasets/<control-dataset>/<dataset>/<energy-or-timing>/<window>/<role>/
  sample_masks/<control-prepared-path>/<mode>/<samples>/
  transforms/<transform>/<control>/<fit-dataset>/<mode>/<window>/seed_<integer>/
  features/<transform>/<control>/<dataset>/<mode>/<window>/seed_<integer>/<role>/<block>/
  imputers/conditional_knn/<control>/<mode>/<window>/w<width>_s<stride>/seed_<integer>/
```

Directory names derive from the explicit dataset label or ROOT filename, board-specific cache root, mode and window. Change the cache root or semantic dataset name when changing preprocessing settings. The cache does not compare configuration hashes to determine whether an existing file should be reused.

A directory is reusable only after the `.complete` marker has been written. Build output is assembled under a temporary sibling directory, then published into its final location. An interrupted build cannot be mistaken for a complete cache. A complete cache can be reused during resume. Concurrent writes to the same semantic cache name are not supported.

## Results and resume

Each run owns `model/`, `artifacts/predictions.npz`, `artifacts/development_predictions.npz`, `artifacts/bootstrap.npz` and its `tables/` and `metadata/` files. The batch planner checks saved outputs and stage completion without a recursive artifact registry.

## Operational rules

- Use separate UC and FBK cache roots and fresh directories for new scientific protocols.
- A changed cutoff, training population, seed or transform configuration requires a new semantic cache path or an explicit rebuild.
- Frozen transforms and sample masks are reused between stages; they are not fitted to Blind.
- Paths inside prepared manifests are relative to the shared preprocessing cache root, so a prepared bundle can be moved between machines.
- Always validate the batch with `check-batch` before launching it, and run the regression tests after modifying the cache implementation.

Historical artifact-store formats are not supported by this branch.
