"""
Centralized configuration for model checkpoint paths.

Fill these paths with your local .pth files before running.
Nothing is loaded at import; consumers should handle missing paths.

Please insert your ViTime model path here
"""

# ——— Inference interfaces used by tools.py ———

VITIME_MODEL_PATH: str | None = 'C:/Users/marce/Documents/Formazione/Projects/Diffusion_Vision_Model_Timeseries/ViTime_Code/ViTime/model/ViTime_Model.pth'
