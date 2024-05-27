from setuptools import setup, find_packages

setup(
    name="multilingual-resource-models",
    version="1.0.0",
    author="omprxkash",
    description="Unified framework for low-resource African NLP: cross-lingual transfer, data augmentation, and knowledge distillation",
    long_description=open("README.md", encoding="utf-8").read() if __import__("os").path.exists("README.md") else "",
    long_description_content_type="text/markdown",
    url="https://github.com/omprxkash/multilingual-resource-models",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.3.0",
        "transformers>=4.40.0",
        "datasets>=2.19.0",
        "tokenizers>=0.19.0",
        "gensim>=4.3.0",
        "scikit-learn>=1.4.0",
        "pandas>=2.2.0",
        "numpy>=1.26.0",
        "tqdm>=4.66.0",
        "pyyaml>=6.0",
        "nltk>=3.8.1",
        "accelerate>=0.29.0",
        "evaluate>=0.4.0",
    ],
    extras_require={
        "dev": ["pytest>=8.0.0", "jupyter>=1.0.0", "matplotlib>=3.8.0", "seaborn>=0.13.0"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
