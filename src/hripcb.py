"""HRIPCB dataset utilities.

The HRIPCB (HRIPCB / PCB_DATASET, Huang et al. 2020) base set is laid out as::

    data/HRIPCB/
        images/<Class>/<id>_<class>_<nn>.jpg
        Annotations/<Class>/<id>_<class>_<nn>.xml   # Pascal-VOC format
        PCB_USED/<id>.JPG                            # 10 defect-free templates

Each image contains one or more defects of a single class; the VOC XML lists one
``<object>`` per defect bounding box.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

# Canonical 6 defect classes (lower-case, as they appear in the XML <name> tag)
CLASSES = [
    "missing_hole",
    "mouse_bite",
    "open_circuit",
    "short",
    "spur",
    "spurious_copper",
]

# English -> Chinese labels for plots / report
CLASS_ZH = {
    "missing_hole": "缺孔",
    "mouse_bite": "鼠咬(缺口)",
    "open_circuit": "斷路",
    "short": "短路",
    "spur": "毛刺(凸尖)",
    "spurious_copper": "多餘銅(殘銅)",
}

# Folder names (capitalised) used by images/ and Annotations/
CLASS_DIRS = [c.capitalize() if "_" not in c else c.title().replace("_", "_")
              for c in CLASSES]


@dataclass
class BBox:
    name: str
    xmin: int
    ymin: int
    xmax: int
    ymax: int

    @property
    def width(self) -> int:
        return self.xmax - self.xmin

    @property
    def height(self) -> int:
        return self.ymax - self.ymin

    @property
    def area(self) -> int:
        return self.width * self.height


@dataclass
class Annotation:
    filename: str
    width: int
    height: int
    depth: int
    objects: list[BBox]
    xml_path: Path

    @property
    def n_objects(self) -> int:
        return len(self.objects)


def parse_voc(xml_path: str | Path) -> Annotation:
    """Parse a single Pascal-VOC XML file into an :class:`Annotation`."""
    xml_path = Path(xml_path)
    root = ET.parse(xml_path).getroot()

    size = root.find("size")
    width = int(size.findtext("width"))
    height = int(size.findtext("height"))
    depth = int(size.findtext("depth"))
    filename = root.findtext("filename")

    objects: list[BBox] = []
    for obj in root.findall("object"):
        name = obj.findtext("name")
        b = obj.find("bndbox")
        objects.append(
            BBox(
                name=name,
                xmin=int(float(b.findtext("xmin"))),
                ymin=int(float(b.findtext("ymin"))),
                xmax=int(float(b.findtext("xmax"))),
                ymax=int(float(b.findtext("ymax"))),
            )
        )
    return Annotation(filename, width, height, depth, objects, xml_path)


def iter_annotations(root: str | Path) -> Iterator[Annotation]:
    """Yield every parsed annotation under ``<root>/Annotations``."""
    ann_dir = Path(root) / "Annotations"
    for xml_path in sorted(ann_dir.rglob("*.xml")):
        yield parse_voc(xml_path)


def image_path_for(ann: Annotation, root: str | Path) -> Path:
    """Resolve the image file that belongs to an annotation.

    The annotation's parent folder name mirrors the images/ sub-folder.
    """
    cls_dir = ann.xml_path.parent.name
    return Path(root) / "images" / cls_dir / ann.filename


def template_paths(root: str | Path) -> list[Path]:
    """Return the PCB_USED defect-free template image paths."""
    pcb_dir = Path(root) / "PCB_USED"
    return sorted(p for p in pcb_dir.iterdir() if p.is_file())


def get_template_for_test(test_filename: str, pcb_used_dir: str | Path) -> str:
    """Map a test image filename to its PCB_USED template path.

    Filenames encode the template id as the leading numeric token, e.g.
    ``01_missing_hole_03.jpg`` -> ``PCB_USED/01.JPG``. The template extension
    casing varies across mirrors, so we match the stem case-insensitively.
    """
    stem = Path(test_filename).name
    tid = stem.split("_")[0]                      # '01'
    pcb_dir = Path(pcb_used_dir)
    for p in pcb_dir.iterdir():
        if p.is_file() and p.stem == tid:
            return str(p)
    raise FileNotFoundError(
        f"no PCB_USED template with id {tid!r} for test image {test_filename!r}"
    )
