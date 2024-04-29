from setuptools import setup
from setuptools.command.build_py import build_py as build_py_orig
from setuptools.command.build_sphinx import build_sphinx

requirements = [
    "numpy",
    "scipy",
]

requirements_dev = ["black", "isort", "flake8", "pre-commit"]


class build_py(build_py_orig):
    def run(self):
        self.run_command("build_sphinx")
        build_py_orig.run(self)


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
    cmdclass={
        "build_py": build_py,
        "build_sphinx": build_sphinx,
    },
    command_options={
        "build_sphinx": {
            "source_dir": ("setup.py", "docs/source"),
            "build_dir": ("setup.py", "docs/html_build"),
        },
    },
)
