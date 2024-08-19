# DataVis2D

DataVis2D is a simple tool for visualizing image data in 2D using pretrained models. The tool allows you to load images from a directory, extract features using a pretrained model (like ResNet), and visualize these features in a 2D scatter plot using dimensionality reduction techniques such as t-SNE or UMAP.

## Features

- Load images from class-separated directories.
- Extract features using pretrained models (ResNet50, ResNet18).
- Visualize image data in 2D with t-SNE or UMAP.
- Simple and intuitive GUI built with Tkinter.

## Requirements

- Python 3.8+
- PyTorch
- torchvision
- numpy
- matplotlib
- scikit-learn
- umap-learn
- Pillow
- tkinter (usually included with Python)

## Installation

First, ensure you have Python 3.8 or higher installed. Then, follow these steps:

1. Clone the repository:

   ```
   git clone https://github.com/yourusername/DataVis2D.git
   cd DataVis2D
   ```

2. Install dependencies using [Poetry](https://python-poetry.org/):

   ```
   poetry install
   ```

   If you encounter any issues with numpy, specify the version explicitly:

   ```
   poetry add numpy@1.25.0
   ```

3. Activate the virtual environment:

   ```
   poetry shell
   ```

## Usage

After installing the dependencies, you can run the application with the following command:

```
python datavis2d/main.py
```

### Example Directory Structure

Your data folder should be structured like this:

```
data/
├── cat/
│   ├── cat.1.jpg
│   ├── cat.2.jpg
│   └── ...
├── dog/
│   ├── dog.1.jpg
│   ├── dog.2.jpg
│   └── ...
```

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
