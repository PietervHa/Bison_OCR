"""
Configuration Loader

This module handles loading and validating the application's configuration from 
YAML files and environment variables. It provides a global `cfg` dictionary 
used throughout the project.
"""

import sys
import argparse
import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv

log = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_PROJECT_ROOT / ".env")


def _warn_missing_env_vars() -> None:
    """
    Checks for required environment variables and issues warnings if they are missing.
    """
    missing = []
    for env_name in ("ROBOFLOW_API_KEY",):
        if not str(os.environ.get(env_name, "")).strip():
            missing.append(env_name)

    if missing:
        log.warning(
            "Missing environment variables: %s. Set them in the project root .env file or your environment.",
            ", ".join(missing),
        )

"""
Configuration loader for the machine vision application.
Loads YAML configuration files and exposes a module-level cfg dictionary.
"""

def _load_config():
    """Load configuration from YAML file (default or CLI override)."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--config", type=str, default=None, help="Path to config YAML file")

    # Parse only known arguments to avoid conflicts with other scripts
    args, _ = parser.parse_known_args()

    # Determine config file path
    if args.config:
        config_path = Path(args.config)
    else:
        config_path = Path(__file__).resolve().parents[2] / "config" / "default.yaml"

    # Check if file exists
    if not config_path.exists():
        log.error("Configuration file not found: %s", config_path)
        sys.exit(1)

    # Load YAML
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
        if config is None:
            config = {}
        return config
    except yaml.YAMLError as e:
        log.error("Failed to parse YAML file %s: %s", config_path, e)
        sys.exit(1)
    except Exception as e:
        log.error("Failed to load config file %s: %s", config_path, e)
        sys.exit(1)


def _as_dict(value):
    """
    Ensures the value is a dictionary, returning an empty dict if not.
    """
    return value if isinstance(value, dict) else {}


def _first_non_empty(*values, default=""):
    """
    Returns the first non-empty value from a list of candidates.
    """
    for value in values:
        if value is None:
            continue
        if isinstance(value, str):
            candidate = value.strip()
            if candidate:
                return candidate
            continue
        if value != "":
            return value
    return default


def _normalize_object_detection_config(config: dict) -> dict:
    """Normalize canonical provider-based object-detection configuration."""
    od_cfg = config.get("object_detection")
    if not isinstance(od_cfg, dict):
        return config

    normalized = dict(od_cfg)
    classifier_cfg = _as_dict(normalized.get("classifier"))
    roboflow_cfg = _as_dict(normalized.get("roboflow"))

    backend = str(_first_non_empty(normalized.get("backend"), default="classifier")).strip().lower()
    if backend not in {"classifier", "roboflow", "yolo", "template"}:
        backend = "classifier"

    normalized["backend"] = backend

    classifier_model_path = str(_first_non_empty(classifier_cfg.get("model_path"))).strip()
    classifier_cfg["model_path"] = classifier_model_path
    normalized["classifier"] = classifier_cfg

    roboflow_model = str(_first_non_empty(roboflow_cfg.get("model"), default="default")).strip()
    roboflow_models = roboflow_cfg.get("models") if isinstance(roboflow_cfg.get("models"), dict) else {}
    selected_profile = _as_dict(roboflow_models.get(roboflow_model))

    roboflow_api_key = str(os.environ.get("ROBOFLOW_API_KEY", "")).strip()
    roboflow_workspace = str(_first_non_empty(selected_profile.get("workspace"))).strip()
    roboflow_workflow = str(_first_non_empty(selected_profile.get("workflow"))).strip()
    roboflow_api_url = str(
        _first_non_empty(selected_profile.get("api_url"), default="https://serverless.roboflow.com")
    ).strip()
    selected_profile["api_key"] = roboflow_api_key
    selected_profile["workspace"] = roboflow_workspace
    selected_profile["workflow"] = roboflow_workflow
    selected_profile["api_url"] = roboflow_api_url

    if not isinstance(roboflow_models, dict):
        roboflow_models = {}
    roboflow_models[roboflow_model] = selected_profile

    normalized["roboflow"] = {
        "model": roboflow_model,
        "models": roboflow_models,
        "api_key": roboflow_api_key,
        "workspace": roboflow_workspace,
        "workflow": roboflow_workflow,
        "api_url": roboflow_api_url,
    }

    # NOTE: maintenance-mode access used to be gated by a single shared
    # password read from here. That's gone, login is now per-user via the
    # `users` table (see backend/core/auth.py, QC_tools/manage_users.py),
    # so there is nothing security-related left to load from config/env.
    config["security"] = _as_dict(config.get("security"))

    config["object_detection"] = normalized
    return config


def _validate_object_detection_config(config: dict) -> None:
    """Warn if Roboflow backend is selected but config is incomplete."""
    od_cfg = config.get("object_detection", {})
    if not isinstance(od_cfg, dict):
        return

    backend = str(od_cfg.get("backend", "classifier")).strip().lower()
    if backend != "roboflow":
        return

    roboflow_cfg = od_cfg.get("roboflow", {})
    if not isinstance(roboflow_cfg, dict):
        log.warning("object_detection.roboflow must be a dict when backend='roboflow'")
        return

    selected_model = str(roboflow_cfg.get("model", "")).strip()
    if not selected_model:
        log.warning("object_detection.roboflow.model is required when backend='roboflow'")
        return

    roboflow_models = roboflow_cfg.get("models", {})
    if not isinstance(roboflow_models, dict):
        log.warning("object_detection.roboflow.models must be a dict when backend='roboflow'")
        return

    if selected_model not in roboflow_models:
        available = list(roboflow_models.keys())
        log.warning(
            f"Roboflow model '{selected_model}' not found in roboflow.models. "
            f"Available models: {available}"
        )
        return

    model_profile = roboflow_models[selected_model]
    if not isinstance(model_profile, dict):
        log.warning("roboflow.models.%s must be a dict", selected_model)
        return

    required_fields = {"api_key", "workspace", "workflow"}
    missing = required_fields - set(model_profile.keys())
    if missing:
        log.warning(
            f"Roboflow model '{selected_model}' profile is incomplete. "
            f"Missing required fields: {missing}. "
            f"All of {required_fields} are required."
        )
        return

    for field in required_fields:
        value = str(model_profile.get(field, "")).strip()
        if not value:
            log.warning(
                f"Roboflow model '{selected_model}' has empty required field '{field}'. "
                f"All required fields must have non-empty values."
            )
            return


# Load configuration at module level
_warn_missing_env_vars()
cfg = _load_config()
cfg = _normalize_object_detection_config(cfg)
_validate_object_detection_config(cfg)

