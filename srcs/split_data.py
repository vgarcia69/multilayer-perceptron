import sys
import pandas as pd
from classes.Dataset import Dataset

def main():
	if len(sys.argv) != 2:
		print(f"Usage: python {sys.argv[0]} <dataset.csv>")
		sys.exit(1)

	try:
		df = pd.read_csv(sys.argv[1], header=None)
	except (FileNotFoundError, pd.errors.EmptyDataError, pd.errors.ParserError) as e:
		print(f"Error: cannot read '{sys.argv[1]}': {e}")
		sys.exit(1)
	dataset = Dataset(df)

	dataset.rand_test_split()

if __name__ == "__main__":
	try:
		main()
	except (EOFError, KeyboardInterrupt):
		print()