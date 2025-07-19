from setuptools import setup, find_packages

setup(
    name="predictive_maintenance",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        # List your dependencies here, or leave empty if using requirements.txt
        "pandas",
        # add others as needed
    ],
)
