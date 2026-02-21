from setuptools import setup, find_packages

setup(
    name="deepRLPH",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "aiohttp>=3.9.1",
        "requests>=2.31.0",
        "python-dotenv>=1.0.0",
        "pydantic>=2.5.0",
        "click>=8.1.7",
        "httpx>=0.24.0",
        "asyncio-contextmanager>=1.0.1",
        "typing-extensions>=4.8.0",
        "numpy>=1.26.4",
        "pandas>=2.2.3",
        "pyarrow==12.0.1",  # Known to have Python 3.9 wheels on macOS
        "onnx==1.14.1",  # Python 3.9 wheels
        "onnxruntime==1.16.3",  # Python 3.9 wheels
    ],
    python_requires=">=3.9",
)
