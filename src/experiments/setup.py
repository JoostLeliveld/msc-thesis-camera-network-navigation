from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'experiments'
here = os.path.abspath(os.path.dirname(__file__))


def package_glob(*parts):
    # Files only: data_files cannot copy directories (e.g. config/archive/), which
    # otherwise breaks the install step.
    return [p for p in glob(os.path.join(*parts), root_dir=here)
            if os.path.isfile(os.path.join(here, p))]

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'),
         package_glob('launch', '*.launch.py')),
        (os.path.join('share', package_name, 'config'),
         package_glob('config', '*')),
        (os.path.join('share', package_name, 'data', 'visibility_gp'),
         package_glob('data', 'visibility_gp', '*')),
    ],
    install_requires=['setuptools', 'PyYAML'],
    zip_safe=True,
    maintainer='joostleliveld',
    maintainer_email='j.j.p.leliveld@student.tue.nl',
    description='Experiment launchers and utilities for pipeline evaluation',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'goal_mission_node = experiments.nodes.goal_mission_node:main',
            'goal_marker_node = experiments.nodes.goal_marker_node:main',
            'experiment_logger = experiments.nodes.experiment_logger:main',
        ],
    },
)
