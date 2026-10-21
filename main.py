"""
CHARGEUP - Application Entrypoint
AI-Powered EV Charging Demand Forecast & Fault Detection
"""

import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from app import app

if __name__ == "__main__":
    import uvicorn
    print("\n=========================================================")
    print("  Starting CHARGEUP Service on http://127.0.0.1:8000     ")
    print("  Interactive Swagger Docs: http://127.0.0.1:8000/docs   ")
    print("=========================================================\n")
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True, app_dir=BASE_DIR)
