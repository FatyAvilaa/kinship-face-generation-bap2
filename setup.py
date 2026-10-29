from setuptools import find_packages, setup

setup(
    name="kinship-face-generation-bap2",
    version="1.0.0",
    description="BAP2: biparental attention with a Heredity Gate to generate children's faces from their parents",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
)
