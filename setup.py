from setuptools import setup, find_packages

setup(
    name="fiba",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "onnxruntime==1.18.0",
        "opencv-python-headless==4.9.0.80",
        "numpy==1.26.4",
        "scipy==1.13.1",
        "spacy==3.7.4",
        "torch==2.3.0",
        "torchvision==0.18.0",
        "huggingface-hub==0.23.4",
        "mediapipe==0.10.33",
        "Pillow==10.3.0",
    ],
    python_requires=">=3.10",
)
