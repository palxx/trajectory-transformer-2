from setuptools import setup, find_packages

setup(
    name="rt_offline",
    version="0.1.0",
    description="Reliability-Guaranteed and Reward-Seeking Transformer (RT) for Model-Based Offline RL",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "torch>=2.6.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "scipy>=1.13.0",
        "scikit-learn>=1.5.0",
    ],
)
