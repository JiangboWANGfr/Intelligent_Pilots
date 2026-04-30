import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.gui.config_gui import main

if __name__ == "__main__":
    config = main()
