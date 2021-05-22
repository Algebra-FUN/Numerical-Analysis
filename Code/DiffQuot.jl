macro diff_quot(expr::Expr)
    func_symbol = expr.args[1].args[1]
    return (quote
            $expr
            $func_symbol(x...)=($func_symbol(x[2:end]...)-$func_symbol(x[1:end-1]...))/(x[end]-x[1])
        end) |> esc
end