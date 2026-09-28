from setuptools import find_packages, setup

package_name = 'unav_common'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools', 'numpy', 'shapely>=1.8'],
    zip_safe=True,
    maintainer='joostleliveld',
    maintainer_email='j.j.p.leliveld@student.tue.nl',
    description='Shared utilities for UnembodiedNavigation',
    license='MIT',
    extras_require={'test': ['pytest']},
)
