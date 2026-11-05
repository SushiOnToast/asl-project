IMG_EXTS = {".jpg", ".jpeg", ".png"}

def find_train_dir(root):
    """Return the folder that directly contains the class folders (A, B, ...)."""
    for p in root.rglob("A"): # recursive search
        if p.is_dir() and (p.parent / "B").is_dir():
            return p.parent
    return None


def list_images(class_dir):
    return sorted(f for f in class_dir.iterdir() if f.suffix.lower() in IMG_EXTS)
