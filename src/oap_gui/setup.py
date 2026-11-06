# Copyright 2026 AI Robot Association (AIRoA)
# SPDX-License-Identifier: Apache-2.0
from setuptools import setup

setup(
    name="oap_gui",
    version="0.1.0",
    packages=["oap_gui"],
    data_files=[("share/ament_index/resource_index/packages", ["resource/oap_gui"]),
                ("share/oap_gui", ["package.xml"])],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="AIRoA",
    maintainer_email="ext-oss@airoa.org",
    description="One window for an OpenArm run: control chain, moves, policy, camera views, bag, overlay video",
    license="Apache-2.0",
    entry_points={"console_scripts": ["gui = oap_gui.app:main", "overlay = oap_gui.overlay:main"]},
)
