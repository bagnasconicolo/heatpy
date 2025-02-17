#!/usr/bin/env python3
import sys
import os
import numpy as np
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QLineEdit, QPushButton, QFileDialog, QMessageBox, QComboBox,
    QCheckBox, QDoubleSpinBox
)
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from PIL import Image

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("2D/3D Intensity Plotter with Multiplot Options")
        self.original_image = None  # Loaded PIL image
        self.image = None           # Processed intensity data (numpy array)
        self.initUI()

    def initUI(self):
        # Main container
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)

        # --- File Selection Layout ---
        file_layout = QHBoxLayout()
        main_layout.addLayout(file_layout)
        file_layout.addWidget(QLabel("Input Image:"))
        self.input_line = QLineEdit()
        file_layout.addWidget(self.input_line)
        input_browse_btn = QPushButton("Browse...")
        file_layout.addWidget(input_browse_btn)
        input_browse_btn.clicked.connect(self.browse_input)

        file_layout.addWidget(QLabel("Output Plot:"))
        self.output_line = QLineEdit()
        file_layout.addWidget(self.output_line)
        output_browse_btn = QPushButton("Browse...")
        file_layout.addWidget(output_browse_btn)
        output_browse_btn.clicked.connect(self.browse_output)

        # --- Plot Options Layout ---
        options_layout = QHBoxLayout()
        main_layout.addLayout(options_layout)

        # Plot Dimension: 2D or 3D
        options_layout.addWidget(QLabel("Plot Dimension:"))
        self.dimension_combo = QComboBox()
        self.dimension_combo.addItems(["2D", "3D"])
        options_layout.addWidget(self.dimension_combo)
        self.dimension_combo.currentTextChanged.connect(self.updatePlotTypeOptions)

        # Plot Type (populated based on dimension)
        options_layout.addWidget(QLabel("Plot Type:"))
        self.plot_type_combo = QComboBox()
        options_layout.addWidget(self.plot_type_combo)
        self.updatePlotTypeOptions(self.dimension_combo.currentText())
        self.plot_type_combo.currentTextChanged.connect(self.update_plot)

        # Global Colormap (for single plot or when separate colormaps not used)
        options_layout.addWidget(QLabel("Global Colormap:"))
        self.colormap_combo = QComboBox()
        self.colormap_combo.addItems(['viridis', 'gray', 'jet', 'plasma', 'inferno'])
        options_layout.addWidget(self.colormap_combo)
        self.colormap_combo.currentTextChanged.connect(self.update_plot)

        # Intensity mode (for single RGB plot or grayscale)
        options_layout.addWidget(QLabel("Intensity Mode:"))
        self.intensity_combo = QComboBox()
        self.intensity_combo.addItems(["Grayscale"])
        self.intensity_combo.setEnabled(False)
        options_layout.addWidget(self.intensity_combo)
        self.intensity_combo.currentTextChanged.connect(self.update_plot)

        # MultiPlot checkbox (only applies to RGB images)
        self.multiplot_checkbox = QCheckBox("MultiPlot for RGB")
        options_layout.addWidget(self.multiplot_checkbox)
        self.multiplot_checkbox.toggled.connect(self.onMultiPlotToggled)
        options_layout.addStretch()

        # --- Multiplot Options Layout ---
        multiplot_layout = QHBoxLayout()
        main_layout.addLayout(multiplot_layout)

        # Subplot spacing adjustment (used when in multiplot mode)
        multiplot_layout.addWidget(QLabel("Subplot Spacing:"))
        self.spacing_spin = QDoubleSpinBox()
        self.spacing_spin.setRange(0.0, 1.0)
        self.spacing_spin.setSingleStep(0.05)
        self.spacing_spin.setValue(0.3)
        multiplot_layout.addWidget(self.spacing_spin)
        self.spacing_spin.valueChanged.connect(self.update_plot)

        # Colorbar position selector
        multiplot_layout.addWidget(QLabel("Colorbar Position:"))
        self.cbpos_combo = QComboBox()
        self.cbpos_combo.addItems(["Right", "Left", "Top", "Bottom"])
        multiplot_layout.addWidget(self.cbpos_combo)
        self.cbpos_combo.currentTextChanged.connect(self.update_plot)

        # Separate colormaps for RGB channels (only for multiplot mode)
        self.separate_cmap_checkbox = QCheckBox("Separate Colormaps for RGB")
        multiplot_layout.addWidget(self.separate_cmap_checkbox)
        self.separate_cmap_checkbox.toggled.connect(self.onSeparateCmapToggled)

        # QComboBoxes for each channel (hidden by default)
        self.r_cmap_combo = QComboBox()
        self.r_cmap_combo.addItems(['viridis', 'gray', 'jet', 'plasma', 'inferno'])
        self.r_cmap_combo.setVisible(False)
        multiplot_layout.addWidget(QLabel("R:"))
        multiplot_layout.addWidget(self.r_cmap_combo)
        self.r_cmap_combo.currentTextChanged.connect(self.update_plot)

        self.g_cmap_combo = QComboBox()
        self.g_cmap_combo.addItems(['viridis', 'gray', 'jet', 'plasma', 'inferno'])
        self.g_cmap_combo.setVisible(False)
        multiplot_layout.addWidget(QLabel("G:"))
        multiplot_layout.addWidget(self.g_cmap_combo)
        self.g_cmap_combo.currentTextChanged.connect(self.update_plot)

        self.b_cmap_combo = QComboBox()
        self.b_cmap_combo.addItems(['viridis', 'gray', 'jet', 'plasma', 'inferno'])
        self.b_cmap_combo.setVisible(False)
        multiplot_layout.addWidget(QLabel("B:"))
        multiplot_layout.addWidget(self.b_cmap_combo)
        self.b_cmap_combo.currentTextChanged.connect(self.update_plot)

        multiplot_layout.addStretch()

        # --- Custom Titles and Axes Labels ---
        labels_layout = QHBoxLayout()
        main_layout.addLayout(labels_layout)
        labels_layout.addWidget(QLabel("Title:"))
        self.title_edit = QLineEdit("Intensity Plot")
        labels_layout.addWidget(self.title_edit)
        labels_layout.addWidget(QLabel("X Label:"))
        self.xlabel_edit = QLineEdit("X")
        labels_layout.addWidget(self.xlabel_edit)
        labels_layout.addWidget(QLabel("Y Label:"))
        self.ylabel_edit = QLineEdit("Y")
        labels_layout.addWidget(self.ylabel_edit)
        labels_layout.addWidget(QLabel("Z Label:"))
        self.zlabel_edit = QLineEdit("Intensity")
        labels_layout.addWidget(self.zlabel_edit)
        labels_layout.addStretch()

        # --- Buttons Layout ---
        buttons_layout = QHBoxLayout()
        main_layout.addLayout(buttons_layout)
        gen_plot_btn = QPushButton("Generate Plot")
        gen_plot_btn.clicked.connect(self.load_and_plot)
        buttons_layout.addWidget(gen_plot_btn)
        save_plot_btn = QPushButton("Save Plot")
        save_plot_btn.clicked.connect(self.save_plot)
        buttons_layout.addWidget(save_plot_btn)
        buttons_layout.addStretch()

        # --- Matplotlib Canvas ---
        self.figure = Figure(figsize=(8, 6))
        self.canvas = FigureCanvas(self.figure)
        main_layout.addWidget(self.canvas)

    def browse_input(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(
            self, "Select Input Image", "",
            "Image Files (*.png *.jpg *.bmp *.tif);;All Files (*)", options=options)
        if file_name:
            self.input_line.setText(file_name)

    def browse_output(self):
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getSaveFileName(
            self, "Select Output Plot File", "",
            "PNG Files (*.png);;JPEG Files (*.jpg);;All Files (*)", options=options)
        if file_name:
            self.output_line.setText(file_name)

    def updatePlotTypeOptions(self, dimension):
        """Update the plot type options based on the selected dimension."""
        self.plot_type_combo.clear()
        if dimension == "2D":
            self.plot_type_combo.addItems(["Image", "Contour"])
        elif dimension == "3D":
            self.plot_type_combo.addItems(["Surface", "Wireframe", "Scatter"])
        self.update_plot()

    def onMultiPlotToggled(self, checked):
        # When MultiPlot is enabled, disable the intensity mode selector.
        if checked:
            self.intensity_combo.setEnabled(False)
        else:
            if self.original_image and self.original_image.mode == 'RGB':
                self.intensity_combo.setEnabled(True)
        self.update_plot()

    def onSeparateCmapToggled(self, checked):
        # Show/hide the separate colormap QComboBoxes.
        self.r_cmap_combo.setVisible(checked)
        self.g_cmap_combo.setVisible(checked)
        self.b_cmap_combo.setVisible(checked)
        self.update_plot()

    def add_colorbar(self, ax, mappable):
        """Add a colorbar to the given axis based on the selected position."""
        pos = self.cbpos_combo.currentText()
        if pos == "Right":
            return self.figure.colorbar(mappable, ax=ax, shrink=0.5, aspect=10)
        elif pos == "Left":
            bbox = ax.get_position()
            cax = self.figure.add_axes([bbox.x0 - 0.05, bbox.y0, 0.03, bbox.height])
            return self.figure.colorbar(mappable, cax=cax, orientation='vertical')
        elif pos == "Top":
            bbox = ax.get_position()
            cax = self.figure.add_axes([bbox.x0, bbox.y1 + 0.01, bbox.width, 0.03])
            return self.figure.colorbar(mappable, cax=cax, orientation='horizontal')
        elif pos == "Bottom":
            bbox = ax.get_position()
            cax = self.figure.add_axes([bbox.x0, bbox.y0 - 0.08, bbox.width, 0.03])
            return self.figure.colorbar(mappable, cax=cax, orientation='horizontal')
        else:
            return self.figure.colorbar(mappable, ax=ax)

    def load_and_plot(self):
        input_path = self.input_line.text().strip()
        if not input_path or not os.path.exists(input_path):
            QMessageBox.warning(self, "Input Error", "Please select a valid input image file.")
            return
        try:
            img = Image.open(input_path)
            # Convert to RGB if not already (to ensure three channels)
            if img.mode not in ['RGB', 'L']:
                img = img.convert('RGB')
            self.original_image = img.copy()  # Store the original image

            # For an RGB image, adjust the intensity mode options
            if img.mode == 'RGB':
                if not self.multiplot_checkbox.isChecked():
                    self.intensity_combo.setEnabled(True)
                    self.intensity_combo.clear()
                    self.intensity_combo.addItems(["Grayscale", "Red", "Green", "Blue"])
                    self.intensity_combo.setCurrentIndex(0)
                else:
                    self.intensity_combo.setEnabled(False)
                # Also store a grayscale version for fallback in single-plot mode
                self.image = np.array(img.convert('L'))
            elif img.mode == 'L':
                self.intensity_combo.setEnabled(False)
                self.image = np.array(img)
            else:
                self.intensity_combo.setEnabled(False)
                self.image = np.array(img.convert('L'))
            self.update_plot()
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load image:\n{str(e)}")

    def update_plot(self):
        if self.original_image is None:
            return

        # Determine if we are in multiplot mode (only for RGB images)
        is_multiplot = self.multiplot_checkbox.isChecked() and self.original_image.mode == 'RGB'
        dimension = self.dimension_combo.currentText()

        # When not in multiplot mode, select one data array based on intensity mode
        if not is_multiplot:
            if self.original_image.mode == 'RGB' and self.intensity_combo.isEnabled():
                mode = self.intensity_combo.currentText() or "Grayscale"
                if mode == "Grayscale":
                    data = np.array(self.original_image.convert('L'))
                elif mode == "Red":
                    data = np.array(self.original_image)[:, :, 0]
                elif mode == "Green":
                    data = np.array(self.original_image)[:, :, 1]
                elif mode == "Blue":
                    data = np.array(self.original_image)[:, :, 2]
                else:
                    data = np.array(self.original_image.convert('L'))
            else:
                data = self.image
        else:
            # For multiplot mode, extract the R, G, and B channels
            rgb_array = np.array(self.original_image)
            data_R = rgb_array[:, :, 0]
            data_G = rgb_array[:, :, 1]
            data_B = rgb_array[:, :, 2]

        # Clear the previous figure
        self.figure.clf()

        # If multiplot mode, adjust subplot spacing
        if is_multiplot:
            spacing = self.spacing_spin.value()
            self.figure.subplots_adjust(wspace=spacing, hspace=spacing)

        # ----- 2D Plots -----
        if dimension == "2D":
            if not is_multiplot:
                ax = self.figure.add_subplot(111)
                plot_type = self.plot_type_combo.currentText()  # "Image" or "Contour"
                if plot_type == "Image":
                    im = ax.imshow(data, cmap=self.colormap_combo.currentText())
                    self.add_colorbar(ax, im)
                elif plot_type == "Contour":
                    rows, cols = data.shape
                    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
                    cont = ax.contourf(X, Y, data, cmap=self.colormap_combo.currentText())
                    self.add_colorbar(ax, cont)
                else:
                    im = ax.imshow(data, cmap=self.colormap_combo.currentText())
                    self.add_colorbar(ax, im)
                ax.set_title(self.title_edit.text())
                ax.set_xlabel(self.xlabel_edit.text())
                ax.set_ylabel(self.ylabel_edit.text())
            else:
                # Multiplot: create a 1x3 layout for R, G, and B
                axes = [self.figure.add_subplot(1, 3, i) for i in range(1, 4)]
                channels = [("Red", data_R), ("Green", data_G), ("Blue", data_B)]
                for ax, (chan, dat) in zip(axes, channels):
                    plot_type = self.plot_type_combo.currentText()
                    # Use separate colormaps if selected
                    if self.separate_cmap_checkbox.isChecked():
                        if chan == "Red":
                            cmap = self.r_cmap_combo.currentText()
                        elif chan == "Green":
                            cmap = self.g_cmap_combo.currentText()
                        elif chan == "Blue":
                            cmap = self.b_cmap_combo.currentText()
                        else:
                            cmap = self.colormap_combo.currentText()
                    else:
                        cmap = self.colormap_combo.currentText()
                    if plot_type == "Image":
                        im = ax.imshow(dat, cmap=cmap)
                        self.add_colorbar(ax, im)
                    elif plot_type == "Contour":
                        rows, cols = dat.shape
                        X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
                        cont = ax.contourf(X, Y, dat, cmap=cmap)
                        self.add_colorbar(ax, cont)
                    else:
                        im = ax.imshow(dat, cmap=cmap)
                        self.add_colorbar(ax, im)
                    ax.set_title(f"{self.title_edit.text()} - {chan}")
                    ax.set_xlabel(self.xlabel_edit.text())
                    ax.set_ylabel(self.ylabel_edit.text())
        # ----- 3D Plots -----
        elif dimension == "3D":
            if not is_multiplot:
                ax = self.figure.add_subplot(111, projection='3d')
                rows, cols = data.shape
                X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
                plot_type = self.plot_type_combo.currentText()  # "Surface", "Wireframe", "Scatter"
                if plot_type == "Surface":
                    surf = ax.plot_surface(X, Y, data.astype(np.float64),
                                           cmap=self.colormap_combo.currentText(),
                                           linewidth=0, antialiased=False)
                    self.add_colorbar(ax, surf)
                elif plot_type == "Wireframe":
                    rstride = max(1, rows // 50)
                    cstride = max(1, cols // 50)
                    ax.plot_wireframe(X, Y, data.astype(np.float64),
                                      color='black', rstride=rstride, cstride=cstride)
                elif plot_type == "Scatter":
                    sc = ax.scatter(X.flatten(), Y.flatten(), data.flatten(),
                                    c=data.flatten(), cmap=self.colormap_combo.currentText(), marker='o')
                    self.add_colorbar(ax, sc)
                else:
                    surf = ax.plot_surface(X, Y, data.astype(np.float64),
                                           cmap=self.colormap_combo.currentText(),
                                           linewidth=0, antialiased=False)
                    self.add_colorbar(ax, surf)
                ax.set_title(self.title_edit.text())
                ax.set_xlabel(self.xlabel_edit.text())
                ax.set_ylabel(self.ylabel_edit.text())
                ax.set_zlabel(self.zlabel_edit.text())
            else:
                # Multiplot 3D: create a 1x3 layout for R, G, and B channels
                axes = [self.figure.add_subplot(1, 3, i, projection='3d') for i in range(1, 4)]
                channels = [("Red", data_R), ("Green", data_G), ("Blue", data_B)]
                for ax, (chan, dat) in zip(axes, channels):
                    rows, cols = dat.shape
                    X, Y = np.meshgrid(np.arange(cols), np.arange(rows))
                    plot_type = self.plot_type_combo.currentText()
                    if self.separate_cmap_checkbox.isChecked():
                        if chan == "Red":
                            cmap = self.r_cmap_combo.currentText()
                        elif chan == "Green":
                            cmap = self.g_cmap_combo.currentText()
                        elif chan == "Blue":
                            cmap = self.b_cmap_combo.currentText()
                        else:
                            cmap = self.colormap_combo.currentText()
                    else:
                        cmap = self.colormap_combo.currentText()
                    if plot_type == "Surface":
                        surf = ax.plot_surface(X, Y, dat.astype(np.float64),
                                               cmap=cmap, linewidth=0, antialiased=False)
                        self.add_colorbar(ax, surf)
                    elif plot_type == "Wireframe":
                        rstride = max(1, rows // 50)
                        cstride = max(1, cols // 50)
                        ax.plot_wireframe(X, Y, dat.astype(np.float64),
                                          color='black', rstride=rstride, cstride=cstride)
                    elif plot_type == "Scatter":
                        sc = ax.scatter(X.flatten(), Y.flatten(), dat.flatten(),
                                        c=dat.flatten(), cmap=cmap, marker='o')
                        self.add_colorbar(ax, sc)
                    else:
                        surf = ax.plot_surface(X, Y, dat.astype(np.float64),
                                               cmap=cmap, linewidth=0, antialiased=False)
                        self.add_colorbar(ax, surf)
                    ax.set_title(f"{self.title_edit.text()} - {chan}")
                    ax.set_xlabel(self.xlabel_edit.text())
                    ax.set_ylabel(self.ylabel_edit.text())
                    ax.set_zlabel(self.zlabel_edit.text())
        self.canvas.draw()

    def save_plot(self):
        output_path = self.output_line.text().strip()
        if not output_path:
            QMessageBox.warning(self, "Output Error", "Please select a valid output file to save the plot.")
            return
        try:
            self.figure.savefig(output_path)
            QMessageBox.information(self, "Success", f"Plot saved to:\n{output_path}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save plot:\n{str(e)}")

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.resize(1200, 900)
    window.show()
    sys.exit(app.exec_())