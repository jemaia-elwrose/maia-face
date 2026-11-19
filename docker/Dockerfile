# CPU image for the default (OpenCV: YuNet + SFace) backend. Runs the broker too:
#   maia-infer broker            maia-infer worker face
FROM python:3.12-slim
ARG EXTRAS=""
ENV PIP_NO_CACHE_DIR=1 PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1 \
    MAIA_BROKER=tcp://broker:5556 MAIA_FACE_HOME=/weights MAIA_FACE_GALLERY=/data/gallery.db
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
# Some extras pull in the desktop OpenCV build, which needs GUI libraries: keep only headless.
RUN pip install ".${EXTRAS:+[$EXTRAS]}" \
 && (pip uninstall -y opencv-python opencv-contrib-python || true) \
 && pip install --force-reinstall --no-deps "opencv-python-headless>=4.8" \
 && python -c "import cv2, maia_face, maia_face.workers; from maia_infer.cli import _load_worker; \
_load_worker('face'); print('ok: maia-face', maia_face.__version__, '| opencv', cv2.__version__)"
CMD ["maia-infer", "worker", "face"]
