#!/usr/bin/env python3
"""
YOLOv8-CFruit 安装脚本
"""

from setuptools import setup, find_packages
import os

# 读取README文件
def read_readme():
    with open('README.md', 'r', encoding='utf-8') as f:
        return f.read()

# 读取requirements文件
def read_requirements():
    with open('requirements.txt', 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name='yolov8-cfruit',
    version='1.0.0',
    author='Your Name',
    author_email='cindynianx@gmail.com',
    description='YOLOv8-CFruit: 专为油茶果检测设计的目标检测模型',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    url='https://github.com/your-username/YOLOv8-CFruit',
    packages=find_packages(),
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    python_requires='>=3.8',
    install_requires=read_requirements(),
    extras_require={
        'dev': [
            'pytest>=7.0.0',
            'black>=22.0.0',
            'flake8>=5.0.0',
            'mypy>=0.991',
        ],
        'docs': [
            'sphinx>=5.0.0',
            'sphinx-rtd-theme>=1.0.0',
        ],
    },
    entry_points={
        'console_scripts': [
            'yolov8-cfruit-train=scripts.train:main',
            'yolov8-cfruit-detect=examples.basic_detection:main',
        ],
    },
    include_package_data=True,
    package_data={
        '': ['*.yaml', '*.yml', '*.json'],
    },
    keywords=[
        'computer-vision',
        'object-detection',
        'yolo',
        'yolov8',
        'agriculture',
        'fruit-detection',
        'camellia-oleifera',
    ],
    project_urls={
        'Bug Reports': 'https://github.com/your-username/YOLOv8-CFruit/issues',
        'Source': 'https://github.com/your-username/YOLOv8-CFruit',
        'Documentation': 'https://github.com/your-username/YOLOv8-CFruit/docs',
    },
) 