"""classes.py - the 30 cobot-relevant ImageNet-1k classes used for the 1000-frame dataset."""

# name -> ImageNet-1k class index (same ordering as timm / torchvision heads)
COBOT_CLASSES: dict[str, int] = {
    "cellular_telephone": 487,
    "computer_keyboard": 508,
    "desktop_computer": 527,
    "forklift": 561,
    "hammer": 587,
    "hatchet": 596,
    "ipod": 605,
    "joystick": 613,
    "laptop": 620,
    "lawn_mower": 621,
    "microphone": 650,
    "microwave": 651,
    "monitor": 664,
    "mouse": 673,
    "padlock": 695,
    "pill_bottle": 720,
    "plane": 726,           # carpenter's plane (the tool)
    "pop_bottle": 737,
    "screen": 782,
    "sewing_machine": 786,
    "shovel": 792,
    "swab": 840,
    "television": 851,
    "toaster": 859,
    "tractor": 866,
    "trailer_truck": 867,
    "waffle_iron": 891,
    "wall_clock": 892,
    "water_bottle": 898,
    "wine_bottle": 907,
}

INDEX_TO_NAME: dict[int, str] = {v: k for k, v in COBOT_CLASSES.items()}


def class_name(idx: int) -> str:
    """Short name for an ImageNet-1k index (cobot names, else timm's description, else the index)."""
    if idx in INDEX_TO_NAME:
        return INDEX_TO_NAME[idx]
    try:
        from timm.data import ImageNetInfo

        return ImageNetInfo().index_to_description(idx).split(",")[0].strip().replace(" ", "_")
    except Exception:
        return f"class_{idx}"
