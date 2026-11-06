# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
from glob import glob

from setuptools import setup

setup(
    name="oap_runner",
    version="0.1.0",
    packages=["oap_runner"],
    data_files=[("share/ament_index/resource_index/packages", ["resource/oap_runner"]),
                ("share/oap_runner", ["package.xml"]),
                ("share/oap_runner/config", glob("config/*.yaml"))],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="AIRoA",
    maintainer_email="ext-oss@airoa.org",
    description="AIRoA-0 policy inference on OpenArm: cameras and state to hand pose chunks and finger commands",
    license="Apache-2.0",
    entry_points={"console_scripts": ["runner = oap_runner.runner:main", "prewarm = oap_runner.prewarm:main"]},
)
