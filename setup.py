from setuptools import setup

requirements = [
    "numpy",
    "scipy",
    "scikit-learn",
    "matplotlib",
    "tqdm",
]

requirements_dev = ["black", "isort", "flake8", "pre-commit"]

setup(
    name="desgld",
    version="0.1.0",
    description="Package for decentralized stochastic gradient descent",
    url="https://github.com/mrislambd/desgld_package.git",
    author="Rafiq Islam",
    packages=["desgld"],
    package_dir={"": "src"},
    install_requires=requirements,
    extras_require={"dev": requirements_dev},
)
