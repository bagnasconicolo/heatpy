# HeatPy
2D/3D Intensity Plotter with Multiplot Options
# 2D/3D Intensity Plotter with Multiplot Options

This Python application provides a graphical user interface (GUI) to visualize image intensity data using 2D or 3D plots. It supports both single plots (using a selected intensity mode) and multiplots for RGB images (displaying separate plots for the Red, Green, and Blue channels). You can customize colormaps, adjust subplot spacing, choose colorbar positions, and set custom titles and axis labels.

## Features

- **2D and 3D Plotting Modes**
  - *2D Plots*: Choose between "Image" and "Contour" representations.
  - *3D Plots*: Choose between "Surface", "Wireframe", or "Scatter" visualizations.
- **Intensity Mode Selector**
  - For RGB images, select Grayscale or a specific channel (Red, Green, Blue) for single plot visualization.
- **RGB MultiPlot Mode**
  - Display three subplots (one for each color channel) for RGB images.
  - Optionally use separate colormaps for each channel.
- **Customizable Layout Options**
  - Adjust subplot spacing.
  - Choose the colorbar position (Right, Left, Top, or Bottom).
- **Customizable Labels and Titles**
  - Set a global plot title and axis labels (X, Y, and Z).
- **Robust File Handling**
  - Browse for an input image file.
  - Specify an output file to save the generated plot.

## Requirements

Make sure you have the following packages installed:

- Python 3.x
- [PyQt5](https://pypi.org/project/PyQt5/)
- [Matplotlib](https://pypi.org/project/matplotlib/)
- [Pillow](https://pypi.org/project/Pillow/)
- [NumPy](https://pypi.org/project/numpy/)

Install the required packages using pip:

```bash
pip install pyqt5 matplotlib pillow numpy
```

 ## Usage
- **Input/Output Files**
  - Use the **Browse** buttons to select the input image file and specify the output file name for saving the plot.
- **Plot Options**
  - **Plot Dimension:** Select “2D” or “3D”.
	- **Plot Type:** Options depend on the chosen dimension:
  - **2D:** “Image” or “Contour”
	- **3D:** “Surface”, “Wireframe”, or “Scatter”
  - **Global Colormap:** Choose a colormap for single plots (or when not using separate colormaps).
	- **Intensity Mode:** For RGB images (in single plot mode), choose between “Grayscale”, “Red”, “Green”, or “Blue”.
- **MultiPlot Mode**
  - Check **MultiPlot for RGB** to display individual subplots for the R, G, and B channels.
	- When enabled, the intensity mode selector is disabled.
- **Multiplot Options**
  - **Subplot Spacing:** Adjust the spacing between subplots using the spin box.
	- **Colorbar Position:** Select where the colorbar should appear (Right, Left, Top, or Bottom).
  - **Separate Colormaps for RGB:** Enable to set individual colormaps for each channel. When checked, additional dropdowns will appear for selecting colormaps for the Red, Green, and Blue channels.
- **Custom Titles and Axes Labels**
  - Modify the text fields to set your desired plot title and labels for the X, Y, and Z axes.
- **Generating and Saving Plots**
  - Click **Generate Plo**t to render the plot on the embedded Matplotlib canvas.
	- Click **Save Plot** to save the current plot to the specified output file.
    
 ## License
This project is licensed under the MIT License.
