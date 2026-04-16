"""
Cache Module
------------
File-based JSON cache with MD5 keys and 24-hour TTL.
"""

import hashlib
import json
import os
import time

# Cache file path (relative to project root)
CACHE_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cache.json")

# Time-to-live in seconds (24 hours)
TTL_SECONDS = 24 * 60 * 60


def _load_cache() -> dict:
    """
    Load cache data from disk.

    Returns:
        Cache dictionary, or empty dict on failure.
    """
    if not os.path.exists(CACHE_FILE):
        return {}

    try:
        with open(CACHE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError) as e:
        print(f"[Cache] Corrupted or unreadable cache file, resetting: {e}")
        return {}


def _save_cache(cache: dict) -> None:
    """
    Save cache data to disk.

    Args:
        cache: Cache dictionary to persist.
    """
    try:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(cache, f, indent=2, ensure_ascii=False)
    except IOError as e:
        print(f"[Cache] Failed to write cache file: {e}")


def _make_key(claim: str) -> str:
    """
    Generate an MD5 hash key from a claim string.

    Args:
        claim: The claim text.

    Returns:
        MD5 hex digest string.
    """
    return hashlib.md5(claim.strip().lower().encode("utf-8")).hexdigest()


def get(claim: str) -> dict | None:
    """
    Retrieve a cached result for a claim.

    Args:
        claim: The claim text to look up.

    Returns:
        Cached result dict if found and not expired, otherwise None.
    """
    cache = _load_cache()
    key = _make_key(claim)

    entry = cache.get(key)
    if entry is None:
        return None

    # Check TTL
    timestamp = entry.get("timestamp", 0)
    if time.time() - timestamp > TTL_SECONDS:
        # Expired — remove entry
        del cache[key]
        _save_cache(cache)
        return None

    return entry.get("result")


def set(claim: str, result: dict) -> None:
    """
    Store a result in the cache.

    Args:
        claim:  The claim text as the cache key.
        result: The result dict to cache.
    """
    cache = _load_cache()
    key = _make_key(claim)

    cache[key] = {
        "timestamp": time.time(),
        "claim": claim.strip(),
        "result": result,
    }

    _save_cache(cache)


def clear() -> None:
    """
    Clear all cached entries.
    """
    _save_cache({})
    print("[Cache] Cache cleared.")
