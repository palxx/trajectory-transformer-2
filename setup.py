from setuptools import setup, find_packages

setup(
    name="rt_offline",
    version="0.1.0",
    description="Reliability-Guaranteed and Reward-Seeking Transformer (RT) for Model-Based Offline RL",
    packages=find_packages(),
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.21.0",
        "tqdm>=4.62.0",
        "scipy>=1.7.0",
        "scikit-learn>=1.0.0",
    ],
)
