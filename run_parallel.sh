#!/bin/sh
parallel --bar --joblog parallel.log --results . 'python examples/wasp107b_real.py {} {}' ::: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 ::: 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15
