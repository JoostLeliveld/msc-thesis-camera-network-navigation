from setuptools import find_packages, setup


package_name = "reliability"

setup(
    name=package_name,
    version="0.0.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        ("share/" + package_name + "/config", ["config/leakage_firewall.yaml"]),
    ],
    install_requires=["setuptools", "pyyaml", "numpy"],
    zip_safe=True,
    maintainer="joostleliveld",
    maintainer_email="j.j.p.leliveld@student.tue.nl",
    description="ROS-independent camera reliability contracts and leakage firewall",
    license="MIT",
    extras_require={"test": ["pytest"]},
    entry_points={
        "console_scripts": [
            "camera_manager_node = reliability.nodes.camera_manager_node:main",
        ],
    },
)
