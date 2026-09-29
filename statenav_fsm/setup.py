#!/usr/bin/env python3
"""Compatibility shim — all metadata lives in pyproject.toml.

This file exists only so `pip install -e .` also works in the older environments this
package has to be installed into (the elevation_mapping and gsa ROS workspaces, Isaac
Sim's bundled python). PEP 660 editable installs need setuptools >= 64; where the build
backend is older, pip falls back to the legacy `setup.py develop` path, which requires
this file to be present. Without it pip refuses outright:

    ERROR: Project ... has a 'pyproject.toml' and its build backend is missing the
    'build_editable' hook. Since it does not have a 'setup.py' nor a 'setup.cfg', it
    cannot be installed in editable mode.

Do not add metadata here — setuptools reads it from pyproject.toml's [project] table.
"""

from setuptools import setup

setup()
