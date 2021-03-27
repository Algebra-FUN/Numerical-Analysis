function matrix2latex(M)
    rows_str = map(i -> join(M[i,:],'&'),1:size(M)[1])
    matrix_str = join(rows_str,"\\\\")
    return "\\begin{bmatrix}$(matrix_str)\\end{bmatrix}"
end

macro latex(M::Symbol)
    symbol = String(M)
    return :( "\$\$\n"*$symbol*"="*matrix2latex($M)*"\n\$\$" |> HTML)
end

macro latex(expr::Expr)
    if expr.head == :(=)
        left = expr.args[1]
        right = expr.args[2] |> eval
        if typeof(left) == Symbol && typeof(right) == Array{Int64,2}
            symbol = String(left)
            expr |> eval
            return :("\$\$\n"*$symbol*"="*matrix2latex($right)*"\n\$\$" |> HTML)
        end
    end
    return expr
end