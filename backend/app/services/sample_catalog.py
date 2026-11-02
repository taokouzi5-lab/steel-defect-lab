"""Server-side whitelist for demo sample images (T3.6).

``manifest.json`` inside the samples directory is the only source of truth
for ``GET /samples`` and ``POST /predict/sample/{id}``. User input is never
joined into filesystem paths: a request can only reference an id present in
the manifest, and manifest entries with path separators or traversal
sequences are rejected at load time.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from pathlib import Path

from app.core.sanitize import sanitize_text

logger = logging.getLogger(__name__)

# Conservative filename/label whitelist: no separators, no traversal.
_COMPONENT_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")


@dataclass(frozen=True)
class SampleInfo:
    """One whitelisted sample: id, filename, and ground-truth label."""

    id: str
    name: str
    label: str


class SampleCatalog:
    """Immutable view of the samples manifest."""

    def __init__(self, samples_dir: Path) -> None:
        self._samples_dir = Path(samples_dir)
        self._samples = self._load()

    def _load(self) -> list[SampleInfo]:
        manifest_path = self._samples_dir / "manifest.json"
        if not manifest_path.is_file():
            logger.warning("samples manifest %s not found; serving no samples", manifest_path)
            return []
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            logger.warning("could not read %s: %s", manifest_path, exc)
            return []

        samples: list[SampleInfo] = []
        seen_ids: set[str] = set()
        for entry in payload.get("samples") or []:
            try:
                info = self._parse_entry(entry)
            except (KeyError, TypeError) as exc:
                logger.warning("skipping malformed samples entry %r: %s", entry, exc)
                continue
            if info is None or info.id in seen_ids:
                if info is not None:
                    logger.warning("skipping duplicate sample id '%s'", sanitize_text(info.id))
                continue
            if not self.path_of(info).is_file():
                logger.warning("sample file %s is missing; skipping", self.path_of(info))
                continue
            seen_ids.add(info.id)
            samples.append(info)
        return samples

    @staticmethod
    def _parse_entry(entry: dict) -> SampleInfo | None:
        sample_id = str(entry["id"])
        name = str(entry["name"])
        label = str(entry["label"])
        # Security: reject anything that could escape the samples directory.
        # Sanitized before logging: manifest content is logged verbatim
        # nowhere (L1 control-character scrubbing).
        if not _COMPONENT_PATTERN.match(sample_id):
            logger.warning("skipping sample with unsafe id %r", sanitize_text(sample_id))
            return None
        if not _COMPONENT_PATTERN.match(name):
            logger.warning("skipping sample with unsafe name %r", sanitize_text(name))
            return None
        if not _COMPONENT_PATTERN.match(label):
            logger.warning("skipping sample with unsafe label %r", sanitize_text(label))
            return None
        return SampleInfo(id=sample_id, name=name, label=label)

    def path_of(self, info: SampleInfo) -> Path:
        return self._samples_dir / info.label / info.name

    def list(self) -> list[SampleInfo]:
        return list(self._samples)

    def get(self, sample_id: str) -> SampleInfo | None:
        for sample in self._samples:
            if sample.id == sample_id:
                return sample
        return None
