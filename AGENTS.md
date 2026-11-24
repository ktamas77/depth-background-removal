# AGENTS.md

An OBS Studio filter for Apple silicon Macs that makes a source transparent behind a cutoff distance,
with depth from Apple's Depth Pro 2 run through Core ML. The README has the installation, the use and
the limits; read it first. The repository is Apple's Depth Pro 2 code (the Python package and the
Core ML exporter, kept as upstream) with the plugin added in `obs-plugin/`.

## Build and check

```bash
make -C obs-plugin install    # builds against /Applications/OBS.app, copies the plugin into OBS's plugin folder
uv run --extra coreml python -m demo.export stream -m depthpro2-s -s 360x640 -w 4    # a model to run
```

Restart OBS, add the filter "Depth Background Removal" to a source and give it the model. OBS's log
has the plugin's lines, prefixed `[depth-background-removal]`: the model and size it loaded, and
after 110 frames, the time the model takes per frame. There are no automated tests; the check is
the filter running in OBS.

## Code

- `obs-plugin/plugin.m`: the whole filter, in Objective-C against the OBS C API.
- `obs-plugin/depth_mask.effect`: the shader that fades the picture behind the cutoff.
- `obs-plugin/Makefile`: fetches the OBS headers, builds, signs ad hoc, installs, zips.
- `.github/workflows/obs-plugin.yml`: builds on every push and pull request, releases on a tag `v*`.
- `src/depth_pro_2/`, `demo/export/`: Apple's Python package and Core ML exporter, unchanged.

## Facts not in the README

- The plugin links against the `libobs` inside `OBS.app` and compiles against the headers of the
  same version, `OBS_VERSION` in the Makefile. OBS loads a plugin built for its own version or an
  older one, so raising `OBS_VERSION` raises the oldest OBS the release works with.
- Threads: OBS calls `render` on its graphics thread, and the model loads and runs on one serial
  queue. `lock` guards what the two share: the model's input size, the frame handed over and the
  inverse depth handed back. `busy` keeps at most one frame with the queue.
- OBS pictures are premultiplied by their alpha, so the shader fades all four channels.
- The network predicts inverse depth normalized by the focal length: depth in meters is
  `f_px / (width * inverse_depth)`. The plugin has no focal length and assumes `f_px / width` of 0.7.
- Weights for the Python package: `checkpoints/<file>` under the current working directory is used
  instead of downloading, with the file names of `PRESETS` and `DepthPro2Config.fov_checkpoint_uri`.
- Export: `torch.export` works on `DepthPro2.forward` (normalized frames padded to a multiple of
  32 in, inverse depth out; one input size per export, and padding more than that changes the
  depth) and on `model.fov.focal_length_px`. For unbuffered mode, start from `CausalStep` in
  `demo/export/stream.py`, one frame per call with its cache kept as model state.
