"""
CHARGEUP - Unsupervised Segmentation & Fault Detection Runner
Executes the K-Means station usage segmentation and proxy fault classification.
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import ml_engine

def main():
    print("[CHARGEUP] Executing Unsupervised Segmentation & Pipeline...")
    ml_engine.run_pipeline()

if __name__ == "__main__":
    main()
