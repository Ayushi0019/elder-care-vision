# motion/__init__.py
"""Motion analysis module — frame differencing, optical flow, DBF, morphology."""
from .motion_detection import MotionDetector
from .optical_flow import OpticalFlowAnalyzer
from .background_filter import DoubleBackgroundFilter
from .morphology import MorphologyProcessor
