# Artifact store architecture

## Purpose

Scientific products are immutable and identified by their effective scientific recipe, ordered parent artifact IDs, implementation identity, and compatibility contract. Report directory names and presentation settings do not belong in scientific identities.

The implementation provides the shared store and integrates it with control preprocessing, event selection, native waveform materialization, prepared datasets, frozen sample masks, frozen feature transforms, feature matrices, and final fitted models. Results hold references to prepared products and model products rather than owning duplicate model weights.

The core primitives are:

- `core/artifact_identity.py`: deterministic recipe normalization and SHA-256 product identities.
- `data/artifact_store.py`: atomic immutable publication, process-level build locks, payload checksums, completion manifests, and read-only NumPy memory maps.
- `data/artifact_registry.py`: explicit source bindings and a rebuildable index. Manifests remain authoritative.

## On-disk layout

```text
<artifact-root>/
  products/<product-type>/<sha256>/
    .artifact.json
    manifest.json  # domain metadata, when required
    <payloads>
  .locks/
  registry.json
  bindings.json
```

The product ID is derived from a canonical recipe. The manifest records the recipe, ordered parents, implementation, compatibility metadata, payload checksums, array shapes/dtypes, and completion status. Products are published from a unique temporary sibling directory; a failed builder leaves no published product. Existing products are never silently overwritten. Corrupt or incomplete products require explicit repair/quarantine.

Use `ArtifactStore.get_or_build(recipe, builder)` to materialize a product and `ArtifactStore.open_array(reference, "array.npy")` to open a read-only memory map. Builders must write all payloads into the supplied temporary directory. Do not put machine-local paths in recipes: bind stable source IDs instead.

## Scientific equivalence policy

The store is persistence infrastructure, not a scientific transformation. Integration must preserve effective inputs, event order, masks, seeds, split assignments, transform state, output dtype/chunking, selected hyperparameters, and prediction values. Recipe identities must include every dependency that can change a product and exclude presentation-only settings.

Migration is a separate, disposable operation. Existing products with uncertain provenance must remain distinct; never refit them to guess an identity. Import and verify checksums, array order/shape/dtype, fitted parameters and representative replay results before switching readers. Retain source files until all consumers resolve canonical references.

## Runtime integration

The runtime uses one configurable artifact root. By default it is `waveform_analysis/artifact_store`; override it for new experiments with `--artifact-root` on `batch` or `check-batch`.

Domain manifests remain inside immutable product directories where they describe scientific payloads. The store completion record is `.artifact.json`, so a product can retain a domain-level `manifest.json` without confusing scientific metadata with storage integrity.

The batch result tree contains compact model references and prepared-product references. Final model weights are stored under `products/final-models`; frozen transforms and feature matrices are shared products. The runtime resolves these references through the artifact store, including diagnostics and TSHAP. Prepared datasets are not modified after publication to attach sample masks; masks are separate immutable products.

Historical result import is not part of this runtime. New experiments should start in a fresh result root.
