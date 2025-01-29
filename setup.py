from pathlib import Path
from setuptools import setup
'''
# Resolve the current directory
directory = Path(__file__).resolve().parent
with open(directory / 'README.md', encoding='utf-8') as f:
    long_description = f.read()
'''
setup(
    name='particle_detection',
    version='1.0.0',
    description='A package for training and evaluating autoencoders for particle detection.',
    author='Your Name',
    license='MIT',
    '''
    long_description=long_description,
    long_description_content_type='text/markdown',
    '''
    packages=[
        'particle_detection',
        'particle_detection.autoencoder',
        'particle_detection.data',
    ],
    package_data={'particle_detection': ['py.typed']},  # Include type hints if using
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    install_requires=[
        "torch",
        "numpy",
        "matplotlib",
        "pillow",
        "opencv-python",
    ],
    python_requires='>=3.7',
    extras_require={
        'testing': [
            "pytest",
            "pytest-xdist",
            "hypothesis",
        ],
        'docs': [
            "mkdocs",
            "mkdocs-material",
        ],
    },
    entry_points={
        "console_scripts": [
            "train_model=particle_detection.autoencoder.train_model:main",
            "evaluate_model=particle_detection.autoencoder.evaluate:main",
        ],
    },
    include_package_data=True,
)

