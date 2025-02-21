from setuptools import setup, find_packages

setup(
    name="GnomieLibrary",
    version="0.1.0",
    packages=find_packages(),
    package_dir={"": "src"}, 
    install_requires=["numpy"],  # Dependencies
    author="Shashank Rao",
    author_email="your.email@example.com",
    description="Package with computer vision and machine learning modules.",
    long_description=open("README.md").read(), # create a README.md file
    long_description_content_type="text/markdown",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License", # change license as needed.
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",  # Minimum Python version
)
