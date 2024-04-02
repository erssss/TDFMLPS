import DF
import numpy as np
import argparse
import time
import pandas as pd
from sklearn.metrics import mean_squared_error

# Handle command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--dataname", help="Input file name", default="GPS")
parser.add_argument("--alg", help="Method name", default="DFDP")
parser.add_argument("--k", type=int, help="The parameter kappa", default=3)
# Parse command-line arguments
args = parser.parse_args()

# Read data from CSV file based on the provided data name
data = pd.read_csv("../data/" + args.dataname + ".csv", index_col=None, header=0)

# Preprocess data
data.iloc[:, 0] = data.iloc[:, 0] - data.iloc[0, 0]
X = np.array(data["time"])
obs = np.array(data[["sensor1", "sensor2", "sensor3", "sensor4"]])
true = np.array(data["true"])

# Get the dimensions of the data
n = obs.shape[0]
m = obs.shape[1]
print("n =", n, "  m =", m)

# ===== Perform the selected algorithm =====

# Algorithm 1 in the paper
if args.alg == "DFDP":
    print("===== start DFDP =====")
    start_time = time.time()
    sel1, _ = DF.DFDP(obs, X, args.k)
    end_time = time.time()
    print("[DFDP] time cost:", end_time - start_time, "loss:", mean_squared_error(true, sel1))

# Algorithm 2 in the paper
elif args.alg == "DFRC":
    print("===== start DFRC =====")
    start_time2 = time.time()
    sel2, _, _ = DF.DFRC(obs, X, args.k)
    end_time2 = time.time()
    print("[DFRC] time cost:", end_time2 - start_time2, "loss:", mean_squared_error(true, sel2))

# Algorithm 3 in the paper
elif args.alg == "DFRT":
    print("===== start DFRT =====")
    start_time3 = time.time()
    sel3 = DF.DFRT(obs, X, args.k)
    end_time3 = time.time()
    print("[DFRT] time cost:", end_time3 - start_time3, "loss:", mean_squared_error(true, sel3))
