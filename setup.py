from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="fake-news-detection",
    version="1.0.0",
    author="MohammadZiyafatAbbas",
    author_email="your.email@example.com",
    description="An advanced fake news detection system using machine learning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MohammadZiyafatAbbas/Fake-News-Detection",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
    ],
    python_requires=">=3.7",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=6.2.4",
            "pytest-cov>=2.12.1",
            "flake8>=3.9.2",
            "black>=21.6b0",
            "isort>=5.9.2",
            "mypy>=0.910",
            "sphinx>=4.1.1",
            "sphinx-rtd-theme>=0.5.2",
        ]
    },
    entry_points={
        "console_scripts": [
            "train-model=src.train_advanced_model:main",
            "run-server=src.app:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.txt", "*.json", "*.yaml", "*.yml"],
        "templates": ["*.html"],
        "static": ["*.css", "*.js"],
    },
)