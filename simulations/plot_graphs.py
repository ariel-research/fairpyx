import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("../experiments-csv/leximin_benchmark.csv", header=None)
df.columns = ["agents", "items", "total_util", "min_util", "runtime"]

plt.figure()
plt.plot(df["agents"], df["runtime"], marker="o")
plt.xlabel("Number of agents/items")
plt.ylabel("Runtime (seconds)")
plt.title("LeximinPrimal – Runtime vs. Problem Size")
plt.grid(True)
plt.savefig("../experiments-csv/runtime_vs_size.png")

plt.figure()
plt.plot(df["agents"], df["total_util"], marker="o")
plt.xlabel("Number of agents/items")
plt.ylabel("Total Utility")
plt.title("LeximinPrimal – Total Utility vs. Problem Size")
plt.grid(True)
plt.savefig("../experiments-csv/total_utility_vs_size.png")

plt.figure()
plt.plot(df["agents"], df["min_util"], marker="o")
plt.xlabel("Number of agents/items")
plt.ylabel("Minimum Utility")
plt.title("LeximinPrimal – Min Utility vs. Problem Size")
plt.grid(True)
plt.savefig("../experiments-csv/min_utility_vs_size.png")
