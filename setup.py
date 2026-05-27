from setuptools import find_packages, setup

setup(
    name="pymss",
    version="2.0.2",
    packages=find_packages(),
    description="Python package for music source separation.",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/pymss-project/pymss",
    author="KitsuneX07",
    maintainer="baicai1145",
    author_email="ghast1085654218@163.com",
    maintainer_email="3423714059@qq.com",
    license="MIT",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Topic :: Multimedia :: Sound/Audio",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Operating System :: OS Independent",
    ],
    keywords="music source separation, audio separation, music processing, machine learning, audio",
    python_requires=">=3.10",
    package_data={
        "pymss": ["resources/model_catalog.json", "resources/vr_modelparams/*.json"],
    },
    install_requires=[
        "av>=14",
        "librosa>=0.10.2",
        "numpy>=1.26",
        "pyyaml>=6.0.1",
        "torch>=2.7.1",
        "tqdm>=4.60",
        "mlx>=0.31.0; sys_platform == 'darwin' and platform_machine == 'arm64'",
    ],
    project_urls={
        "Bug Tracker": "https://github.com/pymss-project/pymss/issues",
        "Source Code": "https://github.com/pymss-project/pymss",
        "Documentation": "https://github.com/pymss-project/pymss/blob/main/README.md",
    },
    entry_points={
        "console_scripts": [
            "pymss=pymss.cli:main",
        ],
    },
)
