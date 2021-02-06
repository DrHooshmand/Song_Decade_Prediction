"""
Shahriar Hooshmand,
CSE5523 final project
Demo
"""

import sys
import time
from pathlib import Path
sys.path.insert(1, '../')
from Source import NN_master, svm, adaboost

def demo():
    # start the timer
    start = time.time() 
    # input file
    inp = "reduced.txt"
    escape = 0
    # output file defined in the script
    file_out = "output.txt"
    NN_master.main(inp, sk= escape, pre="layer_neuron.npy")
    svm.main(inp, sk= escape)
    adaboost.main(inp, sk= escape)
    # stop the timer
    end = time.time()
    print("Elapsed time = ", "%1.4f" % (end - start), "seconds")
    return

if __name__ == "__main__":

    demo()
    print("done")
