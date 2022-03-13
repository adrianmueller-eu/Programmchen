#!/bin/zsh

precision=42
debug=

# args parsing
if [[ $1 == "-d" ]]; then
  debug=true
  shift
fi
args=($@)

input=$args[-1]
# cat > args
if [[ ! -t 0 ]]; then
  input=$(cat)
else
  unset 'args[-1]'
fi
# args > clipboard
if [[ -z "$input" ]]; then
  input=$(pbpaste)
fi

# first utilize bc
if hash bc >&/dev/null; then
  # if the input doesn't contain letters (to prevent "variables" to be resolved to 0)
  # and no modulus (doesn't work with scale != 0)
  if [[ "$input" =~ "base" || ( ! "$input" =~ [[:alpha:]] && ! "$input" =~ % ) ]]; then
    [[ -n $debug ]] && printf "bc: "
    res=$(command bc <<< "scale=$precision;$input" 2>&1 | tr -d '\\\n')
    if [[ "$res" != *error* && "$res" != *"Runtime warning"* ]]; then # assume success
      echo " = $res"; exit
    else
      [[ -n $debug ]] && echo "Error! [$res]"
    fi
  fi
fi

# then ask qalc
if hash qalc >&/dev/null; then
  [[ -n $debug ]] && printf "qalc: "
  res=$(qalc "$input" 2>&1)
  if [[ "$res" != *error* ]]; then
    echo "$res"; exit
  else
    [[ -n $debug ]] && echo "Error! [$res]"
  fi
fi

# then go for wcalc
if hash wcalc >&/dev/null; then
  [[ -n $debug ]] && printf "wcalc: "
  res=$(wcalc --radians -P$precision "${args[@]}" "$input" 2>&1 | sed 's/\.0*$//') # sed 's/\(\.\d*\)0*$/\1/;s/\.0*$//'
  if [[ "$res" != *error* && "$res" != *Inf* && "$res" != *Undefined* ]]; then
    echo "$res"; exit
  else
    [[ -n $debug ]] && echo "Error! [$res]"
  fi
fi

# also try python
if hash python >&/dev/null; then
  [[ -n $debug ]] && printf "python: "
  res=$(python -c "
from math import *  # https://docs.python.org/3/library/math.html
nCr=binom=binomial=comb
nPr=lambda a,b: comb(a,b)*factorial(b)
print($input)" 2>/dev/null)
  if [[ $? == 0 ]]; then
    echo "$res"; exit
  else
    [[ -n $debug ]] && echo "Error! [$res]"
  fi
fi

# and if it still seems unsolvable, ask sage
if hash sage >&/dev/null; then
  # see https://paulmasson.github.io/sagemath-docs/functions
  [[ -n $debug ]] && printf "sage: "
  res=$(sage -c "
nCr=binom=binomial
nPr=lambda a,b: factorial(a)/factorial(a-b)
def Fourier(f, inverse=False):
  var('xi')
  inv = 1 if inverse else -1
  return integral(f*e^(inv*i*2*pi*x*xi),x, -oo, oo)

print($input)")
  if [[ $? == 0 ]]; then
    echo "$res"; exit
  else
    [[ -n $debug ]] && echo "Error! [$res]"
  fi
fi
