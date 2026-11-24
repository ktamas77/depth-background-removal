# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = ["coremltools==9.0", "numpy", "opencv-python"]
# ///
"""The filter without OBS: the webcam in a window, black behind the cutoff distance.

    uv run webcam.py MODEL.mlpackage [CAMERA_INDEX]

The slider sets the cutoff; q or Esc quits.
"""

import sys
import time

import coremltools as ct
import cv2
import numpy as np

# As in plugin.m: no focal length is known, so the cutoff is in approximate meters.
FOCAL_OVER_WIDTH = 0.7
WINDOW = "Depth Background Removal"


def keep(inverse_depth, cutoff):
    """1 nearer than the cutoff, 0 behind it, blended over a 10% band, as depth_mask.effect does."""
    meters = FOCAL_OVER_WIDTH / np.maximum(inverse_depth, 1e-6)
    t = np.clip((meters - 0.95 * cutoff) / (0.1 * cutoff), 0, 1)
    return 1 - t * t * (3 - 2 * t)


assert keep(np.float32(1.4), 1.0) == 1 and keep(np.float32(0.35), 1.0) == 0  # 0.5 m stays, 2 m goes

if len(sys.argv) < 2:
    sys.exit(__doc__)
path = sys.argv[1]
index = int(sys.argv[2]) if len(sys.argv) > 2 else 0
package = ct.models.MLModel(path, skip_model_load=True).get_spec().description
sizes = {function.name: function for function in package.functions}
# A package of several sizes defaults to a slower one than the camera needs.
function = "res_640x360" if "res_640x360" in sizes else package.defaultFunctionName or None
description = sizes.get(function, package)
frame_name, depth_name = description.input[0].name, description.output[0].name
height, width = description.input[0].type.multiArrayType.shape[2:]
model = ct.models.MLModel(path, function_name=function, compute_units=ct.ComputeUnit.CPU_AND_GPU)
state = model.make_state() if description.state else None  # the earlier frames of a temporal window

camera = cv2.VideoCapture(index)
if not camera.isOpened():
    sys.exit(
        f"Cannot open camera {index}. Is it connected, and is this terminal allowed under "
        "System Settings > Privacy & Security > Camera?"
    )
cv2.namedWindow(WINDOW)
cv2.createTrackbar("cutoff, cm", WINDOW, 100, 500, lambda _: None)
patience = time.monotonic() + 5  # a camera can fail its first reads while it starts
# ponytail: the model runs in the loop, so the window shows as many frames a second as the model
# manages. Run the model on a thread, as plugin.m does, if a larger one drags.
while cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) >= 1:
    ok, picture = camera.read()
    if not ok:
        if time.monotonic() > patience:
            sys.exit(f"No picture from camera {index}.")
        cv2.waitKey(10)
        continue
    rgb = cv2.resize(picture, (width, height), interpolation=cv2.INTER_AREA)[:, :, ::-1]
    frame = (rgb.transpose(2, 0, 1)[None] / 255).astype(np.float16)
    inverse_depth = model.predict({frame_name: frame}, state=state)[depth_name]
    cutoff = max(cv2.getTrackbarPos("cutoff, cm", WINDOW), 20) / 100
    mask = keep(inverse_depth.astype(np.float32).reshape(height, width), cutoff)
    mask = cv2.resize(mask, picture.shape[1::-1])
    cv2.imshow(WINDOW, (picture * mask[..., None]).astype(np.uint8))
    if cv2.waitKey(1) in (ord("q"), 27):
        break
