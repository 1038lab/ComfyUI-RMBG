# Bundled model resources

These directories contain configuration, tokenizer vocabulary and model source
files only. Model weights are supplied through absolute file paths in the nodes.
Existing SAM2 YAML configs and SAM3 vocabulary remain in their original locations
under `models/sam2` and `models/sam3`.

Each `sources.json` records the Hugging Face repository, immutable commit,
original download URL, SHA-256 and available license metadata. Upstream license
files and notices are preserved where provided. Mirror repositories do not
always supply license metadata: consult the original model's terms before
redistributing its resources. These files retain their upstream licenses;
the integration's license does not replace them.

Refresh resources explicitly during development/setup:

```sh
python3 scripts/vendor_model_resources.py
```

The script reuses recorded revisions and never downloads weights. Nodes never
invoke it. To update a source revision, edit its manifest deliberately and rerun
the script; inspect the resulting source changes and hashes.

## VRMBG-3.0 access

The upstream repository requires approved access for `config.json`, `model.py`
and `vrmbg3_config.py`. These three files could not be bundled without that
access. Their revision is pinned in `briaai/VRMBG-3.0/sources.json`.
After obtaining access, run the script with `HF_TOKEN` set in your environment,
or install those files from the recorded revision into that directory manually.
Until then VRMBG reports the missing resources; it never falls back to a remote
model or a different local weight file.

## Directory-based loaders

Some libraries require conventional filenames in a model directory. The
integration creates a temporary directory containing copies of these resources
and symlinks to the exact weight files supplied by the user. If the operating
system does not allow symlinks, weights are copied into the temporary directory
instead; this requires additional disk space. User files and repository resources
are never patched in place. Temporary bundles are removed when the process exits.
