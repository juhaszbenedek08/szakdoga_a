from dataclasses import dataclass
from pathlib import Path

from lib.util.anchor_util import TARGETS, EMBEDS

embedders = []


@dataclass
class Bioembedder:
    repr_name: str
    embedder_name: str
    repr_path: Path
    raw_path: Path

    def __post_init__(self):
        embedders.append(self)


seqvec = Bioembedder(
    'target_seqvec',
    'SeqVecEmbedder',
    TARGETS,
    EMBEDS
)

esm = Bioembedder(
    'target_esm',
    'ESM1bEmbedder',
    TARGETS,
    EMBEDS
)

prottrans = Bioembedder(
    'target_prottrans',
    'ProtTransT5XLU50Embedder',
    TARGETS,
    EMBEDS
)
