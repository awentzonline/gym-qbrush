#!/usr/bin/env python
from distutils.core import setup


setup(
    name='gym-qbrush',
    version='0.0.2',
    description='AI Gym environment for drawing images',
    author='Adam Wentz',
    author_email='adam@adamwentz.com',
    url='https://github.com/awentzonline/gym-qbrush',
    packages=[
        'gym_qbrush',
        'gym_qbrush.environments',
    ],
    install_requires=[
        'gym',
        'Pillow',
        'numpy',
        'keras'
    ]
)
