from utils.Expression.expression import Expression

def main():
    exp = Expression.fromLatex(r'( \stack { a \\ b } )')
    print(exp.toLatex())

if __name__ == "__main__":
    main()