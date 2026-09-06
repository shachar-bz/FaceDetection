"""Downloads the models pipeline A needs into this folder's models/ directory.

Run this once, before building the face database. Everything pipeline A uses ships
inside InsightFace's `buffalo_l` pack (SCRFD-10GF detection + 5-point alignment +
ResNet50@WebFace600K recognition), which InsightFace downloads and caches itself the
first time a FaceAnalysis app is prepared - so this script simply triggers that.
"""
from face_pipeline import MODELS_ROOT, MODEL_PACK_NAME, build_face_analysis_app


def main() -> None:
    MODELS_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"Downloading the '{MODEL_PACK_NAME}' model pack into {MODELS_ROOT} (first run only)...")
    build_face_analysis_app()
    print(f"Done. Models are cached under {MODELS_ROOT / 'models' / MODEL_PACK_NAME}")


if __name__ == "__main__":
    main()
