from setuptools import find_packages, setup

setup(
    name="threatsim",
    version="0.2.1",
    description=(
        "Time-series anomaly detection with transformers, Monte Carlo Dropout "
        "uncertainty, and a containerised scoring service"
    ),
    author="Rob Jones",
    packages=find_packages(exclude=["tests", "tests.*"]),
    python_requires=">=3.10",
    install_requires=[
        "torch==2.13.0",
        "pandas==3.0.5",
        "numpy==2.5.2",
        "scikit-learn==1.9.0",
        "matplotlib==3.11.1",
        "seaborn==0.13.2",
    ],
    extras_require={
        # Keep the web stack out of the base install so the research path
        # stays lightweight and `import threatsim` never needs FastAPI.
        "serve": [
            "fastapi==0.141.1",
            "uvicorn[standard]==0.52.4",
            "pydantic==2.13.5",
            "prometheus-client==0.26.0",
        ],
        "dev": ["pytest==9.1.1", "httpx==0.28.1", "jupyter", "jupytext"],
    },
)
