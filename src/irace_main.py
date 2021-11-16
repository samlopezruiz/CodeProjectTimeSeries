import argparse
import gc
import logging
import sys
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from irace.train_test_main import train_test_main


def opt_func(num_encoder_steps, num_heads):
    score = num_encoder_steps ** num_heads
    return score


def main(datfile,
         num_encoder_steps=30,
         num_heads=4,
         hidden_layer_size=160,
         learning_rate=0.01,
         dropout_rate=0.3,
         minibatch_size=64,
         pred_steps=5,
         num_epochs=2,
         ):
    # score = opt_func(num_encoder_steps, num_heads)
    score = train_test_main(num_encoder_steps=num_encoder_steps,
                            num_heads=num_heads,
                            hidden_layer_size=hidden_layer_size,
                            learning_rate=learning_rate,
                            minibatch_size=minibatch_size,
                            dropout_rate=dropout_rate,
                            pred_steps=pred_steps,
                            num_epochs=num_epochs)

    # save the fo values in DATFILE
    with open(datfile, 'w') as f:
        f.write(str(score))

    gc.collect()


if __name__ == "__main__":
    # just check if args are ok
    with open('args.txt', 'w') as f:
        f.write(str(sys.argv))

    # loading example arguments
    ap = argparse.ArgumentParser(description='Hyperparameter selection for TFT Model')
    ap.add_argument("-v", "--verbose", help="increase output verbosity", action="store_true")
    ap.add_argument('--num_encoder_steps', dest='num_encoder_steps', type=int, required=True)
    ap.add_argument('--num_heads', dest='num_heads', type=int, required=True)
    ap.add_argument('--minibatch_size', dest='minibatch_size', type=int, required=True)
    # ap.add_argument('--learning_rate', dest='learning_rate', type=float, required=True)
    ap.add_argument('--hidden_layer_size', dest='hidden_layer_size', type=int, required=True)

    # 1 arg file name to save and load fo value
    ap.add_argument('--datfile', dest='datfile', type=str, required=True,
                    help='File where it will be save the score (result)')

    args = ap.parse_args()
    logging.debug(args)
    # call main function passing args
    main(num_encoder_steps=args.num_encoder_steps,
         num_heads=args.num_heads,
         # learning_rate=args.learning_rate,
         minibatch_size=args.minibatch_size,
         hidden_layer_size=args.hidden_layer_size,
         datfile=args.datfile)
