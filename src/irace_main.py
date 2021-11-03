import argparse
import logging
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from irace.train_test_main import train_test_main


def opt_func(num_encoder_steps, num_heads):
    score = num_encoder_steps ** num_heads
    return score


def main(num_encoder_steps,
         num_heads,
         hidden_layer_size,
         learning_rate,
         minibatch_size,
         DATFILE):

    # score = opt_func(num_encoder_steps, num_heads)
    score = train_test_main(num_encoder_steps=num_encoder_steps,
                            num_heads=num_heads,
                            hidden_layer_size=hidden_layer_size,
                            learning_rate=learning_rate,
                            minibatch_size=minibatch_size)

    # save the fo values in DATFILE
    with open(DATFILE, 'w') as f:
        f.write(str(score))


if __name__ == "__main__":
    # just check if args are ok
    with open('args.txt', 'w') as f:
        f.write(str(sys.argv))

    # loading example arguments
    ap = argparse.ArgumentParser(description='Hyperparameter selection for TFT Model')
    ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    ap.add_argument('--num_encoder_steps', dest='num_encoder_steps', type=int, required=True)
    ap.add_argument('--num_heads', dest='num_heads', type=int, required=True)

    # 1 arg file name to save and load fo value
    ap.add_argument('--datfile', dest='datfile', type=str, required=True,
                    help='File where it will be save the score (result)')

    args = ap.parse_args()
    logging.debug(args)
    # call main function passing args
    main(args.num_encoder_steps, args.num_heads, args.datfile)
