# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
from glob import glob

from setuptools import setup

setup(
    name="oap_planner",
    version="0.1.0",
    packages=["oap_planner"],
    data_files=[("share/ament_index/resource_index/packages", ["resource/oap_planner"]),
                ("share/oap_planner", ["package.xml"]),
                ("share/oap_planner/config", glob("config/*.yaml"))],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="AIRoA",
    maintainer_email="ext-oss@airoa.org",
    description="Ready / park moves for both OpenArm arms, checked against the table and the pillar",
    license="Apache-2.0",
    entry_points={"console_scripts": ["plan = oap_planner.plan:main", "circle = oap_planner.circle:main"]},
)
