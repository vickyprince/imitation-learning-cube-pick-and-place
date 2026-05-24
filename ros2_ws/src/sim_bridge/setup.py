from setuptools import find_packages, setup
import os
from glob import glob

package_name = "sim_bridge"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
        (f"share/{package_name}/launch", glob("launch/*.launch.py")),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="Vicky Prince",
    maintainer_email="vickyprincevictor22@gmail.com",
    description="gym_xarm → ROS2 bridge node",
    license="Apache-2.0",
    entry_points={
        "console_scripts": [
            "xarm_sim_node = sim_bridge.xarm_sim_node:main",
        ],
    },
)
