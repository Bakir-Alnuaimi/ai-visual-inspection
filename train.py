"""Learn what GOOD parts look like (build the memory bank).

Usage:  python train.py --category bottle
"""
import argparse

import patchcore

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--category", default="bottle", help="MVTec AD category, e.g. bottle, screw, grid")
args = parser.parse_args()

print(f"Building memory bank for '{args.category}' ...")
bank = patchcore.build_memory_bank(args.category)
patchcore.save_memory_bank(bank, args.category)
print(f"Done: {bank.shape[0]:,} patches saved to {patchcore.bank_file(args.category)}")
