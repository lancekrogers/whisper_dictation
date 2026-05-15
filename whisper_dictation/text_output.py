"""Text typing: wtype (Linux/Wayland) or pbcopy+osascript (macOS)."""

import json
import logging
import subprocess
import sys
import tomllib
from pathlib import Path

log = logging.getLogger(__name__)

CONFIG_PATH = Path.home() / ".config" / "whisper-dictation" / "config.toml"
IS_MACOS = sys.platform == "darwin"

# Supported paste methods -> wtype args (Linux/Wayland)
PASTE_WTYPE_ARGS = {
    "C-v": ["-M", "ctrl", "v", "-m", "ctrl"],
    "C-S-v": ["-M", "ctrl", "-M", "shift", "v", "-m", "shift", "-m", "ctrl"],
}


def load_paste_methods() -> dict[str, str]:
    """Load paste methods from config file."""
    if not CONFIG_PATH.exists():
        return {}
    try:
        with open(CONFIG_PATH, "rb") as f:
            config = tomllib.load(f)
        return config.get("paste_methods", {})
    except (tomllib.TOMLDecodeError, OSError) as e:
        log.warning(f"Failed to load config: {e}")
        return {}


def get_focused_app_id() -> str | None:
    """Get the app_id of the currently focused window in Sway."""
    if IS_MACOS:
        return None
    try:
        result = subprocess.run(
            ["swaymsg", "-t", "get_tree"],
            capture_output=True, text=True, check=True
        )
        tree = json.loads(result.stdout)

        def find_focused(node):
            if node.get("focused"):
                return node.get("app_id")
            for child in node.get("nodes", []) + node.get("floating_nodes", []):
                found = find_focused(child)
                if found:
                    return found
            return None

        return find_focused(tree)
    except (subprocess.CalledProcessError, json.JSONDecodeError):
        return None


def _type_text_macos(text: str):
    """macOS: copy to clipboard, paste via Cmd+V, restore previous clipboard."""
    old_clipboard = subprocess.run(["pbpaste"], capture_output=True).stdout
    subprocess.run(["pbcopy"], input=text.encode(), check=True)
    subprocess.run(
        [
            "osascript",
            "-e",
            'tell application "System Events" to keystroke "v" using command down',
        ],
        check=False,
    )
    subprocess.run(["pbcopy"], input=old_clipboard, check=True)
    log.info(f"Pasted {len(text)} chars (macOS)")


def _type_text_linux(text: str):
    """Linux/Wayland: wtype directly, or wl-copy + paste keychord for known apps."""
    app_id = get_focused_app_id()
    paste_methods = load_paste_methods()
    method = paste_methods.get(app_id) if app_id else None
    paste_args = PASTE_WTYPE_ARGS.get(method) if method else None

    if paste_args:
        # Save clipboard, paste, restore
        old_clipboard = subprocess.run(
            ["wl-paste", "--no-newline"], capture_output=True
        ).stdout
        subprocess.run(["wl-copy", "--", text], check=True)
        subprocess.run(["wtype"] + paste_args, check=False)
        subprocess.run(["wl-copy", "--"], input=old_clipboard, check=True)
        log.info(f"Pasted {len(text)} chars via {method} ({app_id})")
    else:
        subprocess.run(["wtype", "--", text], check=False)
        log.info(f"Typed {len(text)} chars ({app_id or 'unknown'})")


def type_text(text: str):
    """Type text using the right primitive for the current OS."""
    if IS_MACOS:
        _type_text_macos(text)
    else:
        _type_text_linux(text)
