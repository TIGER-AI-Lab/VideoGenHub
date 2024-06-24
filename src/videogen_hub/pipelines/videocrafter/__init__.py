import os
import sys
from pathlib import Path

cur_path = str(Path(__file__).parent.absolute())
sys.path.insert(0, cur_path)
sys.path.insert(0, os.path.join(cur_path, 'lvdm'))
