"""Start every child process without a console window on Windows.

Importing this module patches :class:`subprocess.Popen` so ffmpeg/ffprobe and
every other child inherit ``CREATE_NO_WINDOW`` and a hidden startup window.
Without it, running SimpleVox from a hidden parent (pythonw, a scheduled task,
a watcher) flashes a black console window for each of the 100+ ffmpeg calls in
one episode.

Import it once, before any subprocess use::

    import _nowindow  # noqa: F401

The patch is idempotent and a no-op off Windows.
"""

import os
import subprocess

CREATE_NO_WINDOW = 0x08000000
STARTF_USESHOWWINDOW = 0x00000001
SW_HIDE = 0


def _install() -> None:
    if os.name != "nt":
        return
    if getattr(subprocess.Popen, "_simplevox_hidden", False):
        return

    _base_popen = subprocess.Popen

    class _HiddenPopen(_base_popen):  # type: ignore[misc, valid-type]
        """Popen that never allocates a visible console window."""

        _simplevox_hidden = True

        def __init__(self, *args, **kwargs):
            kwargs["creationflags"] = (
                int(kwargs.get("creationflags", 0)) | CREATE_NO_WINDOW
            )
            startupinfo = kwargs.get("startupinfo")
            if startupinfo is None:
                startupinfo = subprocess.STARTUPINFO()
                kwargs["startupinfo"] = startupinfo
            startupinfo.dwFlags |= STARTF_USESHOWWINDOW
            startupinfo.wShowWindow = SW_HIDE
            super().__init__(*args, **kwargs)

    subprocess.Popen = _HiddenPopen  # type: ignore[assignment]


_install()
