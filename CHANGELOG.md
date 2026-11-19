# Changelog

## 0.1.3
- `face-attr`: a missing checkpoint now lists the checkpoints that are there and how to pick one.

## 0.1.2
- `associate()` accepts detector rows with extra columns (YOLO's `(N, 6)`) directly.
- `examples/age_gender.py`: YOLO people + faces + MiVOLO, end to end.

## 0.1.1
- MiVOLO: `maia-face setup-mivolo` fetches its source at a pinned commit (its package can't
  be built with current setuptools); `[mivolo]` extra pins timm 0.8.13.dev0, the version its
  model code works with. Missing crops follow MiVOLO's own handling; face-only checkpoints
  supported. New `Dockerfile.attributes` and compose profile `attributes`.
- Silence OpenCV 5's harmless "new graph engine" warnings while loading the face models.

## 0.1.0
- First release: YuNet + SFace (default, commercial-friendly) and InsightFace backends,
  shared 5-point alignment, SQLite identity gallery (multi-process safe), IdentityVoter and
  AttributeSmoother, face-person association, MiVOLO age/gender worker, `face` and
  `face-attr` maia-infer workers, CLI, Docker.
