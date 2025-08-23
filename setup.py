"""
Setup configuration for MAILMIND2.O
"""

from setuptools import setup, find_packages
import os

# Read README for long description
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

# Read requirements
with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

# Package metadata
setup(
    name="mailmind2o",
    version="2.0.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="Smart Email Prioritizer & Gmail Assistant powered by AI",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mailmind2.o",
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/mailmind2.o/issues",
        "Documentation": "https://github.com/yourusername/mailmind2.o/docs",
        "Source Code": "https://github.com/yourusername/mailmind2.o",
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: End Users/Desktop",
        "Intended Audience :: Developers",
        "Topic :: Communications :: Email",
        "Topic :: Office/Business",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(
        exclude=["tests*", "docs*", "scripts*"]
    ),
    python_requires=">=3.11",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.3",
            "pytest-asyncio>=0.21.1",
            "pytest-cov>=4.1.0",
            "black>=23.11.0",
            "flake8>=6.1.0",
            "isort>=5.12.0",
            "mypy>=1.7.1",
        ],
        "docs": [
            "sphinx>=7.1.2",
            "sphinx-rtd-theme>=1.3.0",
            "myst-parser>=2.0.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "mailmind2o=app.main:main",
            "mailmind2o-setup=scripts.setup_auth:main",
            "mailmind2o-test=scripts.test_connections:main",
        ],
    },
    include_package_data=True,
    package_data={
        "": [
            "config/*.yaml",
            "config/prompts/*.yaml",
            "config/settings/*.yaml",
            "static/*",
            "templates/*",
        ],
    },
    zip_safe=False,
    keywords=[
        "email",
        "gmail",
        "ai",
        "prioritization",
        "automation",
        "assistant",
        "streamlit",
        "langchain",
        "groq",
    ],
)