#!/bin/bash -v

# compile
g++ *.cpp -o ../knapsack || exit 1
echo "Compilation succeeded!"

# test
../knapsack --help >/dev/null || exit 1

out=$(../knapsack test.dat | grep "Solution" | grep -o "[0-9]*")
[[ $out > $((141278 - 1000)) ]] || exit 1

echo "Tests succeeded!"
