"""Setup file for SAN - Syntax-Aware Network for Handwritten Mathematical Expression Recognition"""

from setuptools import setup, find_packages

setup(
    name="san-hmer",
    version="1.0.0",
    description="Syntax-Aware Network for Handwritten Mathematical Expression Recognition",
    author="SAN Contributors",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.6.0",
        "torchvision>=0.8.0",
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "Pillow>=8.0.0",
        "PyYAML>=5.3.0",
        "tensorboardX>=2.1",
        "tqdm>=4.50.0",
        "flask>=2.0.0",
        "flask-cors>=3.0.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "black>=21.0",
            "flake8>=3.8.0",
        ]
    },
    entry_points={
        "console_scripts": [
            "san-train=training.train:main",
            "san-infer=inference.inference:main",
            "san-server=app.backend.server:main",
        ],
    },
)