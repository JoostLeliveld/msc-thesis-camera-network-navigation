from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'sim'
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(BASE_DIR)
data_files = [
    ('share/ament_index/resource_index/packages',
        ['resource/' + package_name]),
    ('share/' + package_name, ['package.xml']),

    # launch files
    (os.path.join('share', package_name, 'launch'),
        glob('launch/*.launch.py')),
]

# install robot_description files
for root, _, files in os.walk('robot_description'):
    data_files.append(
        (os.path.join('share', package_name, root),
         [os.path.join(root, f) for f in files])
    )

# install gazebo_worlds files
for root, _, files in os.walk('gazebo_worlds'):
    data_files.append(
        (os.path.join('share', package_name, root),
         [os.path.join(root, f) for f in files])
    )

# install models files
for root, _, files in os.walk('models'):
    data_files.append(
        (os.path.join('share', package_name, root),
         [os.path.join(root, f) for f in files])
    )

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=data_files,
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='joostleliveld',
    maintainer_email='j.j.p.leliveld@student.tue.nl',
    entry_points={
        'console_scripts': [
            'actuation_noise_node = sim.actuation_noise_node:main',
            'clock_throttle_node = sim.clock_throttle_node:main',
            'encoder_noise_node = sim.encoder_noise_node:main',
            'wait_for_clock = sim.wait_for_clock:main',
            'wait_for_odom = sim.wait_for_odom:main',
            'reset_world = sim.reset_world:main',
        ],
    },
)
