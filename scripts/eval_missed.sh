#!/bin/bash
echo "all-sgd-2"
./eval_all.sh all-sgd-2 > all-sgd-2.txt
echo "modern-sgd-2"
./eval_all.sh modern-sgd-2 > modern-sgd-2.txt
echo "legacy-sgd-2"
./eval_all.sh legacy-sgd-2 > legacy-sgd-2.txt
echo "all-raw-2"
./eval_all.sh all-raw-2 > all-raw-2.txt
echo "modern-raw-2"
./eval_all.sh modern-raw-2 > modern-raw-2.txt
echo "legacy-raw-2"
./eval_all.sh legacy-raw-2 > legacy-raw-2.txt
