import sys
import pandas as pd
from classes.Dataset import Dataset

def main():
	df = pd.read_csv("data/data.csv", header=None)
	if df is None:
		sys.exit(1)

	dataset = Dataset(df)

	dataset.rand_test_split()

if __name__ == "__main__":
	try:
		main()
	except (EOFError, KeyboardInterrupt):
		print()