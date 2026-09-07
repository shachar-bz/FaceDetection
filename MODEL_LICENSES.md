# Model and third-party license notice

This repository does not commit model-weight files. Its download command obtains pretrained
models from their upstream publishers. Source-code licenses and model-weight licenses are
separate, and users are responsible for confirming that their intended use is permitted.

## InsightFace `buffalo_l`

Both Pipeline A and Pipeline B use the SCRFD detector from InsightFace's `buffalo_l` model
pack. Pipeline A also uses the pack's ResNet50@WebFace600K recognition model.

InsightFace states that its Python source code is MIT-licensed, but its provided pretrained
models—including automatically and manually downloaded models—are available for
**non-commercial research purposes only**. Commercial use requires separate authorization
from InsightFace or replacement with appropriately licensed weights.

- Upstream license statement: <https://github.com/deepinsight/insightface/tree/master/python-package#license>
- Commercial licensing information: <https://www.insightface.ai/>

## OpenCV Zoo SFace

Pipeline B downloads `face_recognition_sface_2021dec.onnx` from the OpenCV Zoo SFace model
directory. That directory includes the Apache License 2.0.

- Model directory: <https://github.com/opencv/opencv_zoo/tree/main/models/face_recognition_sface>
- License file: <https://github.com/opencv/opencv_zoo/blob/main/models/face_recognition_sface/LICENSE>

## Python dependencies

Packages installed from `requirements.txt` remain subject to their respective upstream
licenses. Installing a dependency does not change the license of this project's source code.

## This project's source code

This file documents third-party terms; it does not grant a license to this repository's own
source code. The repository owner must select and add a project source-code license separately
before others have general permission to copy, modify, or redistribute that code.

This notice is informational and is not legal advice.
