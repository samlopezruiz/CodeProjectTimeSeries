import argparse
import logging
import sys

def opt_func(num_encoder_steps, num_heads):
	# score = MUTPB * POP / 100
	# score = float(score)
	# score = score - float(CXPB)
	# if score < 0:
	# 	score = 0

	score = num_encoder_steps ** num_heads
	return score

def main(num_encoder_steps, num_heads, DATFILE):

	score = opt_func(num_encoder_steps, num_heads)

	# save the fo values in DATFILE
	with open(DATFILE, 'w') as f:
		f.write(str(score*100))

if __name__ == "__main__":
	# just check if args are ok
	with open('args.txt', 'w') as f:
		f.write(str(sys.argv))
	
	# loading example arguments
	ap = argparse.ArgumentParser(description='Feature Selection using GA with DecisionTreeClassifier')
	ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
	# 3 args to test values
	ap.add_argument('--num_encoder_steps', dest='num_encoder_steps', type=int, required=True, help='Population size')
	ap.add_argument('--num_heads', dest='num_heads', type=float, required=True, help='Crossover probability')
	# ap.add_argument('--mut', dest='mut', type=float, required=True, help='Mutation probability')
	# 1 arg file name to save and load fo value
	ap.add_argument('--datfile', dest='datfile', type=str, required=True, help='File where it will be save the score (result)')

	args = ap.parse_args()
	logging.debug(args)
	# call main function passing args
	main(args.num_encoder_steps, args.num_heads, args.datfile)