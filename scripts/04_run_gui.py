# -*- coding: utf-8 -*-
"""脚本4：启动图形界面。"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.gui.app import main

if __name__ == "__main__":
    main()
