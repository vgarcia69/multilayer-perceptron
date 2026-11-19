import os
import sys
from utils.io_helper import load_csv
from classes.Dataset import Dataset

def main():
	if len(sys.argv) != 2:
		print(f"Usage: python {sys.argv[0]} <dataset.csv>")
		sys.exit(1)

	df = load_csv(sys.argv[1])
	dataset = Dataset(df)

	try:
		os.makedirs("data", exist_ok=True)
		dataset.rand_test_split()
	except OSError as e:
		print(f"Error: cannot write split files: {e}", file=sys.stderr)
		sys.exit(1)

if __name__ == "__main__":
	try:
		main()
	except (EOFError, KeyboardInterrupt):
		print()