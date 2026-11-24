"""
Product profiles (config/profiles/<name>.yaml).

A profile changes how the code is found and read for one kind of product:
the print formats, the locator and ink settings, the checks and the font
reader model - everything under `ocr:`. Its `ocr:` section is merged over
the loaded config file (config/default.yaml or config/production.yaml):
mappings key by key, any other value (a list such as `templates`, a number,
a string) replaced as a whole. A profile cannot change anything outside
`ocr:` - camera, PLC trigger, web, output stay as the config file has them,
so the PLC sees the same behaviour whatever profile is active.

    label: "Blue bottle"          # shown in the HMI dropdown (default: the name)
    description: "..."            # optional, shown as the dropdown's tooltip
    ocr:
      ink_channel: {...}
      read_validation: {font_reader: {model: ...}}

"standard" is the config file as it is (config/profiles/standard.yaml has no
`ocr:` overrides; it also works when that file is missing). The HMI switches
profiles at runtime (backend/core/vision.py, switch_profile); every result
records the profile it was read with ("profile").
"""
from __future__ import annotations

import copy
import re
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROFILE_DIR = PROJECT_ROOT / "config" / "profiles"
STANDARD = "standard"
_NAME = re.compile(r"^[A-Za-z0-9_-]{1,40}$")
_ALLOWED_KEYS = {"label", "description", "ocr"}


class ProfileError(ValueError):
    """A profile that does not exist or cannot be used."""


def names(directory=None) -> list:
    """Profile names: "standard" first, then the other <name>.yaml files sorted."""
    d = Path(directory or PROFILE_DIR)
    found = sorted(p.stem for p in d.glob("*.yaml") if _NAME.match(p.stem)) if d.is_dir() else []
    return [STANDARD] + [n for n in found if n != STANDARD]


def load(name, directory=None) -> dict:
    """The profile file as a dict (label, description, ocr). Raises
    ProfileError for an invalid name, a missing file (except "standard"), a
    file that is not a mapping, or keys outside label / description / ocr."""
    if not isinstance(name, str) or not _NAME.match(name):
        raise ProfileError(f"invalid profile name {name!r} (letters, digits, - and _ only)")
    path = Path(directory or PROFILE_DIR) / f"{name}.yaml"
    if not path.exists():
        if name == STANDARD:
            return {"label": "Standard", "ocr": {}}
        raise ProfileError(f"profile {name!r} not found ({path})")
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ProfileError(f"profile {name!r}: invalid YAML: {exc}") from None
    if not isinstance(data, dict):
        raise ProfileError(f"profile {name!r}: must be a mapping")
    extra = sorted(set(data) - _ALLOWED_KEYS)
    if extra:
        raise ProfileError(f"profile {name!r}: only label, description and ocr may be set, not {extra}")
    if not isinstance(data.get("ocr") or {}, dict):
        raise ProfileError(f"profile {name!r}: ocr must be a mapping")
    data["ocr"] = data.get("ocr") or {}
    return data


def merge(base, over):
    """base with over merged in, as a new dict: mappings merged key by key,
    every other value replaced. Neither argument is changed."""
    out = copy.deepcopy(base)
    for key, value in (over or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def config_for(name, base=None, directory=None) -> dict:
    """The whole config with profile `name`'s ocr overrides merged in (a copy;
    base defaults to the loaded config file)."""
    if base is None:
        from backend.core.config_loader import cfg as base
    data = load(name, directory)
    out = copy.deepcopy(base)
    out["ocr"] = merge(base.get("ocr") or {}, data["ocr"])
    return out


def describe(name, directory=None) -> dict:
    """{"name", "label", "description"} for the HMI; a profile file that
    cannot be loaded is still listed, with the error as its description."""
    try:
        data = load(name, directory)
        return {"name": name, "label": str(data.get("label") or name),
                "description": str(data.get("description") or "")}
    except ProfileError as exc:
        return {"name": name, "label": f"{name} (invalid)", "description": str(exc)}
