from setuptools import find_packages, setup

setup(
    name="kinship-face-generation-bap2",
    version="1.0.0",
    description="BAP2: atención biparental con Heredity Gate para generar rostros de hijos a partir de sus padres",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
)
