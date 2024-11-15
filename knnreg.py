import numpy as np
import pandas as pd


class MyKNNReg:
    def __init__(self, k = 3):
        self.k = k
        
        
    def __repr__(self):
        return f'MyKNNReg class: k={self.k}'