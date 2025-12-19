from setuptools import setup, find_packages

setup(
    name="stochastic-spike-generator",
    version="0.1.0",
    description="Computational neuroscience project on spike train generation",
    author="Your Name",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pandas>=1.3.0",
        "pytest>=6.2.0",
    ],
)