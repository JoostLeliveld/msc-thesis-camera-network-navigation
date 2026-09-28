from setuptools import find_packages, setup

package_name = 'perception'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/tf_static.launch.py']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='joostleliveld',
    maintainer_email='j.j.p.leliveld@student.tue.nl',
    description='YOLO external-camera perception for UnembodiedNavigation',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'batched_four_camera_yolo_node = perception.nodes.batched_four_camera_yolo_node:main',
            'yolo_robot_detector_node = perception.nodes.yolo_robot_detector_node:main',
            'scheduled_camera_detector_node = perception.nodes.scheduled_camera_detector_node:main',
        ],
    },

)
