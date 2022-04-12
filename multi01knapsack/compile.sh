#!/bin/bash -v

die() {
  echo "ERROR: $*"
  exit 1
}

# compile
g++ *.cpp -o ../knapsack || exit 1
echo "Compilation succeeded!"

# test
../knapsack --help >/dev/null || die "Help failed"

out=$(../knapsack test.dat | grep -i "Solution" | grep -o "[0-9]*")
[[ $out > $((141278 - 1000)) ]] || die "Bad solution"

echo "Tests succeeded!"
