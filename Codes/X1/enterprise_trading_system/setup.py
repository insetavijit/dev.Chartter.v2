from setuptools import setup, find_packages

setup(
    name="enterprise_trading_system",
    version="1.0.0",
    author="Trading Systems Team",
    author_email="your.email@example.com",  # Update this
    description="A comprehensive trading system for generating signals, backtesting, and visualization.",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/enterprise_trading_system",  # Optional: GitHub repo
    packages=find_packages(),
    install_requires=[
        # Core dependencies from your script
        "pandas>=1.3.0",
        "numpy>=1.21.0",
        "mplfinance>=0.12.0",
        "matplotlib>=3.5.0",
        # Add custom ones if they are pip-installable; otherwise, document them
        # "bktst", "resampler", "tafm", "ChartterX5",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    entry_points={
        "console_scripts": [
            "run-trading-analysis = enterprise_trading_system.orchestrator:main",  # If you add a main() function
        ]
    },
)
