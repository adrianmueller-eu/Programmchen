from sys import argv, exit
from . import solve, table, modelCnt, cnf

def main():

    if len(argv) == 1 or "-h" in argv or "--help" in argv:
        print("A SAT solver.")
        print("Usage: \033[1msat\033[0m [-h | -t | -tt | -c | --cnf] \033[4mformula\033[0m")
        print("\t\033[1m-t\033[0m, --table         Show the whole truth table.")
        print("\t\033[1m-tt\033[0m, --table-true   Show the rows of the truth table where \033[4mformula\033[0m is satiesfied")
        print("\t\033[1m-c\033[0m, --count         Count the world in which \033[4mformula\033[0m is satisfied.")
        print("\t\033[1m--cnf\033[0m               Output \033[4mformula\033[0m in CNF.")
        print("\t--dpll              Use python dpll solver instead of z3-solver.")
        print("")
        print("Examples:")
        print("\tsat \"a and b\"")
        print("\tsat \"(~a | b) & (a -> c) & (b -> ~c) & a\"")
        print("\tsat -t \"a <-> b and a -> ~b\"")
        print("\tsat -tt \"(a -> b) & (b <-> c) <-> (b & ~c)\"")
        print("\tsat -c \"a <- not b and not c and b\"")
        print("\tsat --cnf \"r and not q or not t\"")
        exit()

    def arg(s):
        try:
            argv.remove(s)
            return True
        except:
            return False

    useDPLL = arg("--dpll")
    showTable = arg("-t") or arg("--table")
    showCnf = arg("--cnf")
    showTableTrueOnly = arg("-tt") or arg("--table-true")
    showModelCnt = arg("-c") or arg("--count")

    if len(argv) > 2:
        print("Invalid arguments. See --help for more help.")
        exit(1)
    elif len(argv) == 1:
        print("Please specify a formula.")
        exit(1)

    if showTable and showModelCnt:
        print("Please specify only one output type.")
        exit(1)

    formula = argv[1]

    try:
        if showTable or showTableTrueOnly:
            table(formula, trueOnly=showTableTrueOnly)
        elif showModelCnt:
            print(modelCnt(formula))
        elif showCnf:
            cnf(formula, verbose=True)
        else:
            result, model = solve(formula, useDPLL)

            if result:
                print("Satisfiable with model")
                print(model)
            else:
                print("Unsatisfiable")
    except ValueError as e:
        print(e)
        exit(1)

if __name__ == '__main__':
    main()
