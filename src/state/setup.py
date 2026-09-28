from setuptools import find_packages, setup

package_name = 'state'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
         ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='joostleliveld',
    maintainer_email='j.j.p.leliveld@student.tue.nl',
    description='State adapter providing /state/bev for external-observer experiments.',
    license='Apache-2.0',
    extras_require={
        'test': ['pytest'],
    },
    entry_points={
        'console_scripts': [
            'pixel_to_bev_state_node = state.nodes.pixel_to_bev_state_node:main',
        ],
    },
)
