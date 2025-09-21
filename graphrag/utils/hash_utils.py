import hashlib
from ..config.settings import get_settings

settings = get_settings()

SUPPORTED = {"sha256": hashlib.sha256, "md5": hashlib.md5}

def hash_text(text: str) -> str:
    algo = settings.hash_algo.lower()
    hfunc = SUPPORTED.get(algo, hashlib.sha256)
    return hfunc(text.encode('utf-8', errors='ignore')).hexdigest()
