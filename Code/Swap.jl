macro swap(expr::Expr)
    if expr.head == :tuple
        expr¹ = expr |> deepcopy
        expr¹.args |> reverse!
        return :($expr=$expr¹) |> esc
    end
end