"""
motion/morphology.py — Morphological Operations
================================================
PURPOSE:
    Clean up the foreground mask produced by motion detection and DBF.

    After thresholding, the foreground mask often has:
    - Small isolated white specks (noise)
    - Gaps/holes inside the human body region
    - Thin disconnected blobs

    Morphological operations fix these issues.

WHAT ARE MORPHOLOGICAL OPERATIONS?
    They are shape-based image processing operations.
    A "structuring element" (kernel) slides over the image and
    modifies each pixel based on its neighborhood.

THE FOUR OPERATIONS USED:

    EROSION:
        Shrinks white regions.
        Removes tiny white specks (isolated noise pixels).
        If a white pixel doesn't have enough white neighbors → it becomes black.
        Use case: Remove noise dots.
        
    DILATION:
        Expands white regions.
        Fills small gaps and connects nearby blobs.
        If a black pixel has enough white neighbors → it becomes white.
        Use case: Connect parts of body that are slightly separated.
        
    OPENING = EROSION first, then DILATION:
        Removes small noise while keeping larger structures intact.
        Use case: Clean up noise without destroying the human body blob.
        
    CLOSING = DILATION first, then EROSION:
        Fills holes inside white regions.
        Use case: Fill gaps inside the human body silhouette.

TYPICAL PIPELINE:
    Raw mask (noisy) 
    → Opening  (remove noise specks)
    → Closing  (fill body gaps)
    → Dilation (strengthen/connect body region)
    → Clean mask (ready for pose estimation)
"""

import cv2
import numpy as np
import logging
from typing import Tuple
import config

logger = logging.getLogger(__name__)


class MorphologyProcessor:
    """
    Applies morphological operations to clean the foreground mask.

    Works on binary masks (0 or 255 per pixel) produced by motion
    detection and Double Background Filtering.
    """

    def __init__(
        self,
        kernel_size: Tuple[int, int] = config.MORPH_KERNEL_SIZE,
        dilation_iterations: int = config.MORPH_DILATION_ITER,
    ):
        """
        Args:
            kernel_size:         Size of structuring element, e.g., (5,5)
                                 Larger = more aggressive operation.
            dilation_iterations: How many times to apply dilation.
                                 More iterations = larger dilation.
        """
        self.kernel_size = kernel_size
        self.dilation_iterations = dilation_iterations

        # Create structuring element (rectangular kernel)
        # MORPH_RECT = rectangular shape for the kernel
        # You could also use MORPH_ELLIPSE for a softer effect.
        self.kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, kernel_size
        )

        logger.info(
            f"MorphologyProcessor: kernel={kernel_size}, "
            f"dilation_iter={dilation_iterations}"
        )

    def opening(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological OPENING: erosion then dilation.

        Purpose: Remove small noise specks while keeping large shapes.

        A tiny noise blob (smaller than kernel) is completely removed.
        A large blob (the human body) survives because after erosion
        it's smaller but still there, then dilation restores its size.

        Args:
            mask: Binary mask (0 or 255 per pixel)

        Returns:
            Mask with small noise removed.
        """
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, self.kernel)

    def closing(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological CLOSING: dilation then erosion.

        Purpose: Fill holes and gaps inside foreground regions.

        The human body might appear as multiple blobs with small gaps
        between limbs. Closing connects them into a more solid region.

        Args:
            mask: Binary mask (0 or 255 per pixel)

        Returns:
            Mask with holes filled.
        """
        return cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.kernel)

    def dilation(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply morphological DILATION.

        Purpose: Expand and strengthen the foreground region.

        After opening and closing, we dilate to make the body region
        slightly larger — this helps MediaPipe pose estimation because
        it works better when the person's silhouette is clearly visible.

        Args:
            mask: Binary mask (0 or 255 per pixel)

        Returns:
            Expanded mask.
        """
        return cv2.dilate(mask, self.kernel, iterations=self.dilation_iterations)

    def process(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply the full morphological processing pipeline.

        Pipeline:
            1. Opening  → remove small noise
            2. Closing  → fill body gaps
            3. Dilation → strengthen connections

        Args:
            mask: Raw binary foreground mask (0 or 255)

        Returns:
            Cleaned binary mask ready for pose estimation.
        """
        if mask is None or mask.size == 0:
            return mask

        # Step 1: Opening — noise removal
        # Before: many tiny white specks scattered around
        # After:  only larger connected regions remain
        opened = self.opening(mask)

        # Step 2: Closing — gap filling
        # Before: body might have dark holes inside
        # After:  body is more solid and connected
        closed = self.closing(opened)

        # Step 3: Dilation — region strengthening
        # Before: body region is tight
        # After:  body region is slightly larger, more visible
        dilated = self.dilation(closed)

        white_before = int(np.count_nonzero(mask))
        white_after = int(np.count_nonzero(dilated))

        logger.debug(
            f"Morphology: foreground pixels {white_before} → {white_after}"
        )

        return dilated

    def get_contours(self, mask: np.ndarray):
        """
        Find contours (outlines) of foreground regions in the clean mask.

        Contours can be used to:
        - Estimate bounding box of the person
        - Filter small irrelevant regions
        - Identify the largest (human body) region

        Args:
            mask: Cleaned binary mask

        Returns:
            List of contours (each contour is an array of points).
        """
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return contours

    def largest_contour_area(self, mask: np.ndarray) -> float:
        """
        Get the area of the largest contour in the mask.

        Useful to check if there's a large enough foreground region
        (indicating a person) before attempting pose estimation.

        Args:
            mask: Cleaned binary mask

        Returns:
            Area in pixels of the largest contour. 0 if no contours.
        """
        contours = self.get_contours(mask)
        if not contours:
            return 0.0
        areas = [cv2.contourArea(c) for c in contours]
        return float(max(areas))
