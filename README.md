# Depth Background Removal

A third-party filter plugin for [OBS Studio](https://obsproject.com) on Apple silicon Macs. It removes the background of a camera, or of any other source, by distance: everything farther than a cutoff becomes transparent. The distance comes from Apple's [Depth Pro 2](https://github.com/apple-aiml-research/ml-depth-pro-2), a model that estimates the depth of every pixel from an ordinary camera picture, run on the Mac with Core ML.

A filter that looks for people keeps people. This one keeps whatever is near: you, and also the thing you hold up to the camera.

This is an independent project: it is not part of OBS or made by the OBS Project, and it is not an Apple project. The repository is a copy of Apple's Depth Pro 2 code with the plugin added in [`plugin/`](plugin). The Python package stays because it exports the models the plugin runs.

[Depth RGB Trails](https://github.com/ktamas77/depth-rgb-trails) is a second filter on the same model: it keeps the background and trails the foreground with red, blue and green delayed copies.

## Requirements

- A Mac with Apple silicon, on macOS 15 or later
- OBS Studio 33 or later

## Install

1. Download `depth-background-removal-macos-arm64.zip` from the [latest release](https://github.com/ktamas77/depth-background-removal/releases/latest) and unzip it.
2. Move `depth-background-removal.plugin` into `~/Library/Application Support/obs-studio/plugins/`, creating the folder if it is missing.
3. The plugin is not notarized, and macOS marks downloads as quarantined, so it may refuse to let OBS load it. Clear the mark:

   ```bash
   xattr -dr com.apple.quarantine ~/Library/Application\ Support/obs-studio/plugins/depth-background-removal.plugin
   ```

4. Restart OBS.

Release 0.1.0 was named `obs-depth-background-removal.plugin`. If you installed it, delete that one: OBS would otherwise find the same filter twice.

## Get a model

The plugin ships without a model. Apple publishes exported ones on [Hugging Face](https://huggingface.co/apple/DepthPro2-coreml); with [uv](https://docs.astral.sh/uv/), this downloads the smallest, S (51 MB), into `~/Models`:

```bash
uvx --from huggingface_hub hf download apple/DepthPro2-coreml --include "depthpro2-s_w4.mlpackage/*" --local-dir ~/Models
```

The package holds five input sizes, from 640x360 to 1280x720, which the filter chooses between. To export a model at another size, or with another temporal window, see [demo/README.md](demo/README.md).

## Use

1. In OBS, right-click a source, choose **Filters**, and under Effect Filters add **Depth Background Removal**.
2. **Model**: choose the `.mlpackage`. If the file dialog opens the package as a folder, choose the `Manifest.json` inside it.
3. **Model input size**: start with 640x360. A larger size gives a finer edge and costs speed.
4. **Cutoff distance**: move it until you are kept and the room is gone.

The source is now transparent behind the cutoff, so whatever is below it in the scene shows through.

## Without OBS

[`webcam.py`](webcam.py) makes the same cut in a window of its own: the webcam, black behind the cutoff. It needs the model and [uv](https://docs.astral.sh/uv/), which installs what the script uses, and no OBS:

```bash
uv run webcam.py ~/Models/depthpro2-s_w4.mlpackage
```

The slider sets the cutoff in centimeters, and `q` or Esc quits. A number after the model chooses another camera; `0` is the default. From a package of several sizes the script takes 640x360.

The window is a preview, not a camera that other apps can choose. The model runs once for every frame shown, so the window shows as many frames per second as the model manages.

## Speed

The picture always keeps the source's size and OBS's frame rate. The model runs beside it and the mask follows as fast as the model allows. After its first 110 frames the plugin writes the model's time per frame to OBS's log (Help > Log Files > View Current Log), in a line that starts with `[depth-background-removal]`.

Measured in OBS 33.0.3 on a MacBook Pro with an M1 Max, with a 1280x720 camera on a 1920x1080 canvas at 30 frames per second:

| Model | Input size | Model time per frame | OBS |
|---|---|---|---|
| S, window 4 | 640x360 | about 34 ms (28 to 41 over eight runs) | 30 fps, 1.6 to 1.8 ms per rendered frame |

So on that Mac the mask follows at 25 to 30 frames per second, depending on what else the machine is doing.

The same model at 854x480 took 70 ms per frame on that Mac in a test outside OBS, so the mask would follow at about 14 frames per second. [demo/README.md](demo/README.md) has Apple's timings of every model and size on an M4 Max.

## Limits

- The cutoff is in approximate meters. An OBS source carries no focal length, so the plugin assumes a lens about 71 degrees wide.
- The cut is one distance. Anything else nearer than the cutoff, such as the edge of a desk, stays in the picture.
- The mask is a frame or two behind the picture, which can show at the edge during fast motion.
- The source is stretched to the model's 16:9 input. For a source of another shape, export a model of that shape.
- Tested on one machine: an M1 Max with OBS 33.0.3 and its OpenGL renderer. HDR sources and OBS's Metal renderer are untested, and there is no build for Intel Macs.

## Build from source

With Xcode's command line tools and OBS installed in `/Applications`:

```bash
make -C plugin install
```

This fetches the headers of the OBS version named in [`plugin/Makefile`](plugin/Makefile), builds the plugin against the `libobs` inside `OBS.app`, and copies it into OBS's plugin folder. Restart OBS to load it.

## Releases

GitHub Actions builds the plugin on every push and pull request to `main`; the zip is in the run's artifacts. Pushing a tag `v*` also publishes that zip as a release.

## The rest of the repository

`src/depth_pro_2` and `demo/export` are Apple's code, unchanged: the Depth Pro 2 Python package, its `depth-pro-2 predict` command for photos and videos, and the Core ML exporter. The [upstream README](https://github.com/apple-aiml-research/ml-depth-pro-2#readme) documents them. The Mac demo app is not included here.

## Citation

The model is from this paper:

```bibtex
@article{DepthPro2:2026:arxiv,
  title      = {Depth Pro 2: Sharp and Temporally Consistent Metric Monocular Depth},
  author     = {Peiyun Hu and Marko Mihajlovic and Lyndon Duong and Lahav Lipson and Wei Dong and Ama\"{e}l Delaunoy and Stephan R. Richter and Vladlen Koltun},
  journal    = {arXiv preprint arXiv:2611.26603},
  year       = {2026},
  url        = {https://arxiv.org/abs/2611.26603},
}
```

## License

- The plugin, everything in [`plugin/`](plugin), is under the GNU General Public License, version 2 or later ([plugin/LICENSE](plugin/LICENSE)), as OBS is.
- Apple's code is under [LICENSE](LICENSE), with the third-party code it builds on listed in [ACKNOWLEDGEMENTS](ACKNOWLEDGEMENTS).
- The model weights are under [LICENSE_MODEL](LICENSE_MODEL), which permits use "exclusively for Research Purposes", defined as "non-commercial scientific research and academic development activities". The plugin includes no weights; whether your use of the model fits that license is yours to check.
