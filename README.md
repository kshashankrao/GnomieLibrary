# GnomieLibrary

## Description
GnomieLibrary is a personal learning project containing a collection of Python implementations for computer vision, machine learning, and robotics. It includes basic implementations of algorithms like Particle Filters, DBSCAN clustering, Self-Attention, Inverse Perspective Mapping (IPM), and 3D coordinate transformations. 

## Installation

To install the library locally so that it can be imported cleanly into your other projects, run the following commands:

```bash
cd GnomieLibrary
pip install -e .
```

You can then import modules from anywhere on your system, for example: `from gnomie_library import ParticleFilter`.

## Documentation (MkDocs)

This repository includes a beautiful, fully searchable documentation website powered by **MkDocs** and the **Material** theme. The documentation contains copy-pasteable usage examples for every module.

### How to view the Documentation

1. **Install Dependencies**: Ensure you have installed the required dependencies from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```
   *(This installs `mkdocs` and `mkdocs-material` alongside your other packages)*

2. **Start the Local Server**: Run the following command in the root of the project:
   ```bash
   mkdocs serve
   ```

3. **Browse the Docs**: Open the URL provided in your terminal (usually `http://127.0.0.1:8000`) in your web browser. You will see a modern website with a search bar at the top!

If you prefer, you can also read the raw markdown examples directly in the `docs/` directory.

## Testing

The library includes automated unit tests for all core modules. To run the tests, execute the following command from the root of the project:

```bash
python -m unittest discover tests/
```
