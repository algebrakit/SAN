from utils.Expression.expression import Expression

def main():
    # latex = r'x _ { 2 } ^ { 3 }'
    latex = r'\overline { S } _ { l } ^ { k }'
    expr = Expression.fromLatex(latex)
    print(latex)
    print(expr.toLatex())
    print(expr.to_hybrid())

if __name__ == "__main__":
    main()