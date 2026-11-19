"""maia-face: face detection, recognition, attributes and identity gallery for video analytics."""

from .align import align_face
from .association import associate, crop
from .gallery import Gallery, Match
from .tracking import AttributeSmoother, IdentityVoter, TrackDecision
from .types import Face, pack_faces, unpack_faces

__all__ = [
    "AttributeSmoother",
    "Face",
    "Gallery",
    "IdentityVoter",
    "Match",
    "TrackDecision",
    "align_face",
    "associate",
    "crop",
    "pack_faces",
    "unpack_faces",
]
__version__ = "0.1.0"
