# pipeline_a_resnet50_webface600k

Standalone face-identification pipeline: ResNet50@WebFace600K (InsightFace buffalo_l), TOP3 @ threshold 0.30.

Point it at a folder of labelled photos of the people you care about, then hand it any new
image and it tells you which of those people appear in it. Anyone not in your folder is
reported as `unknown`.

Generated from the FaceDetection project. See that repository for how this configuration was
chosen; this folder is a copy, so edit it there and regenerate rather than editing here.

## Setup

```bash
pip install -r requirements.txt
python -m face_identity.model_downloads --group pipelines
```

## Build the database of people you want recognised

Lay your images out with one folder per person:

```
my_people/
  Person One/  photo1.jpg  photo2.jpg  photo3.jpg
  Person Two/  photo1.jpg  photo2.jpg  photo3.jpg
```

Three to five clear, varied photos per person works well.

```bash
python build_face_database.py --people-images-root my_people
```

## Identify people in a new image

```bash
python identify_faces.py path/to/photo.jpg
python identify_faces.py path/to/photo.jpg --annotated-output labelled.jpg
```

## Configuration

Everything tunable -- the embedding model, the matching strategy and the decision threshold --
lives in `face_identity/configuration.py`. This bundle defaults to `a`:

| Setting | Value |
|---|---|
| Embedding model | `resnet_webface600k` |
| Matching strategy | `top3` |
| Decision threshold | 0.3 |
| Detection confidence | 0.5 |

Raising the threshold means fewer wrong names but more people reported `unknown`; lowering it
does the reverse.
