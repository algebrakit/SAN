from utils.Expression.expression import Expression

def main():
    latex = r'3 \lognl [ 2 ] ( x )'
    expr = Expression.fromLatex(latex)
    print(latex)
    print(expr.toLatex())
    print(expr.to_hybrid())

if __name__ == "__main__":
    main()