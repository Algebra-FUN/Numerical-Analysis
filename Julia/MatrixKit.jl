using LambdaFn
import Markdown

eye(n) = Matrix{Float64}(I,n,n)

function matrix2latex(M)
    rows_str = @λ(join(M[_,:],'&')) |> @λ map(_,1:size(M)[1])
    matrix_str = join(rows_str,"\\\\")
    return "\\begin{bmatrix}$(matrix_str)\\end{bmatrix}"
end

function matrix_form(symbol::Symbol,M)
    return "<center>``"*String(symbol)*"="*matrix2latex(M)*"``</center>" |> Markdown.parse
end

macro latex(M::Symbol)
    return matrix_form(M,M |> eval)
end

macro latex(expr::Expr)
    if expr.head == :(=)
        left = expr.args[1]
        right = expr.args[2] |> eval
        if typeof(left) == Symbol && typeof(right) in (Array{Float64,2},Vector{Float64})
            expr |> eval
            return matrix_form(left,right)
        end
    end
    return expr
end