from setuptools import setup, find_packages

setup(
    name="spinstep",
    version="0.1.0",
    description="Quaternion-based tree traversal for orientation-aware structures.",
    author="Eraldo Marques",
    author_email="eraldo.bernardo@gmail.com",
    license="MIT",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.22",
        "scipy>=1.8"
    ],
    python_requires=">=3.8",
    include_package_data=True,
)
