#!/usr/bin/env python3
import io
import os
import re
from glob import glob
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

install_requires = _load_requirements(os.path.dirname(os.path.realpath(__file__)))

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
        (os.path.join('share', 'statenav_global', 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', 'statenav_global', 'rviz'), glob('ROS/*.rviz')),
    ],
    entry_points={
        'console_scripts': [
            'traversability_estimation = executables.main_worldmodel:main',
            'global_planning = executables.main_global_planning:main',
            'main_multiprocess = executables.main_multiprocess:main',
        ],
    },
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        "dev": ["isort", "black", "pyright"],
        "test": ["pytest"],
    },
    license="MIT",
    zip_safe=True,
    classifiers=[
        "Programming Language :: Python :: 3",
    ],
)
