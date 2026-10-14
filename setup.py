#!/usr/bin/env python3
import io
import os
import re
from typing import List

from setuptools import find_packages, setup


def read(*names, **kwargs):
    with io.open(
        os.path.join(os.path.dirname(__file__), *names),
        encoding=kwargs.get("encoding", "utf8"),
    ) as fp:
        return fp.read()


def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]", version_file, re.M)
    if version_match:
        return version_match.group(1)
    raise RuntimeError("Unable to find version string.")


def _load_requirements(
    path_dir: str, file_name: str = "requirements.txt", comment_char: str = "#"
) -> List[str]:
    """Load requirements from a file
    >>> _load_requirements(PROJECT_ROOT)  # doctest: +ELLIPSIS +NORMALIZE_WHITESPACE
    ['numpy...', 'torch...', ...]
    """
    with open(os.path.join(path_dir, file_name), "r") as file:
        lines = [ln.strip() for ln in file.readlines()]
    reqs = []
    for ln in lines:
        # filer all comments
        if comment_char in ln:
            ln = ln[: ln.index(comment_char)].strip()
        # skip directly installed dependencies
        if ln.startswith("http"):
            continue
        if ln:  # if requirement is not empty
            reqs.append(ln)
    return reqs


readme_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "README.md")
readme = open(readme_path).read() if os.path.exists(readme_path) else ""
version = find_version("statenav_global", "__init__.py")

# Base install = the numpy-level contract layer (statenav_global.core). Every module is opt-in via
# an extra, e.g. `pip install -e .[worldmodel]`. requirements.txt
# stays the full Docker image list (docker/startup.sh installs it explicitly before this package).
# torch / torchvision are left unpinned: install the CUDA (or CPU) build for your platform first,
# then pip keeps it. Tested versions of everything else: constraints.txt (pip install -c ...).
install_requires = ["numpy<2", "omegaconf"]
_WORLDMODEL_REQUIRES = ["torch", "torchvision", "scipy", "matplotlib", "hydra-core>=1.1", "transforms3d"]
extras_require = {
    "worldmodel": _WORLDMODEL_REQUIRES,                     # statenav_global.world_model / utility / inference
    "planners": _WORLDMODEL_REQUIRES + ["pillow"],          # statenav_global.planners (A*)
    "mppi": ["torch"],                                      # statenav_mppi (MPPI local planner)
    "dev": ["isort", "black", "pyright"],
    "test": ["pytest"],
}
extras_require["all"] = sorted(set(sum((extras_require[k] for k in ("worldmodel", "planners", "mppi")), [])))

setup(
    name="statenav_global",
    version=version,
    author="Ziwon Yoon",
    author_email="zyoon6@gatech.edu",
    description="STATE-NAV Global Mapping and Planning: A Python Library for Global Navigation",
    long_description=readme,
    long_description_content_type="text/markdown",
    url="https://github.com/zyoon6/TBA",
    packages=find_packages(exclude=["test", "test.*"]),
    package_data={
        "statenav_global": ["configs/*.yaml"],
    },
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + 'statenav_global']),
        ('share/statenav_global', ['package.xml']),
    ],
    entry_points={
        'console_scripts': [
            'main_worldmodel = executables.main_worldmodel:main',
            'main_fsm = executables.main_fsm:main',
            'main_global_planning = executables.main_global_planning:main',
            'main_multiprocess = executables.main_multiprocess:main',
            'main_mppi = executables.main_mppi:main',
        ],
    },
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require=extras_require,
    license="MIT",
    zip_safe=True,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)