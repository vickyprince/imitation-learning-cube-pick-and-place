from setuptools import find_packages, setup
package_name = "teleop_bridge"
setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", [f"resource/{package_name}"]),
        (f"share/{package_name}", ["package.xml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    entry_points={
        "console_scripts": [
            "teleop_node=teleop_bridge.teleop_node:main",
            "recording_manager=teleop_bridge.recording_manager:main",
            "training_manager=teleop_bridge.training_manager:main",
        ],
    },
)
