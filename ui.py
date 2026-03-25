"""
Colab UI module for SAR Oil Spill Detection.
Enables one-line UI creation with 'import ui'.
"""
import ipywidgets as widgets
from IPython.display import display, clear_output
import matplotlib.pyplot as plt
from PIL import Image
from inference import analyze_sar
import os

# Create Output Area
out = widgets.Output()

# Create Input Box
image_path_input = widgets.Text(
    value='dataset/Oil_spill/34_815,35_138_2.jpg',
    placeholder='Paste image path (e.g., dataset/Oil_spill/image.jpg)',
    description='Image Path:',
    style={'description_width': 'initial'},
    layout=widgets.Layout(width='600px')
)

# Create Analyze Button
analyze_button = widgets.Button(
    description='🚀 Analyze SAR Image',
    button_style='success',
    tooltip='Click to run AI analysis',
    icon='search'
)

def on_button_clicked(b):
    with out:
        clear_output()
        path = image_path_input.value
        
        if not os.path.exists(path):
            print(f"❌ Error: Image path not found: {path}")
            return
            
        try:
            print(f"⏳ Analyzing: {path}...")
            results = analyze_sar(path)
            
            # Show Results Summary
            print(f"\n✅ Prediction: {results['class_name']}")
            print(f"✅ Confidence: {results['confidence']}%")
            print(f"✅ Estimated Area: {results['area_km2']} km²")
            print(f"----------------------------------------\n")

            # Visualize Heatmap and Contour Boundary
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            heatmap_img = Image.open(results['heatmap_path'])
            ax1.imshow(heatmap_img)
            ax1.set_title(f"Grad-CAM Heatmap", fontsize=12, fontweight='bold')
            ax1.axis('off')
            
            contour_img = Image.open(results['contour_path'])
            ax2.imshow(contour_img)
            ax2.set_title(f"Contour Boundary", fontsize=12, fontweight='bold')
            ax2.axis('off')
            
            plt.tight_layout()
            plt.show()

        except Exception as e:
            print(f"❌ Error during analysis: {str(e)}")

analyze_button.on_click(on_button_clicked)

# Automatically display the UI when imported
print("✨ SAR Oil Spill Analyzer UI Loaded")
display(widgets.VBox([image_path_input, analyze_button, out]))
