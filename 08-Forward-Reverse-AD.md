Here’s your full document converted to **GitHub-flavored Markdown**. All LaTeX has been preserved using backticks and code blocks where appropriate. TikZ figures and LaTeX environments like `table`, `minted`, and `align` have been replaced with equivalent Markdown formatting or code blocks where possible, though visual diagrams like TikZ should ideally be replaced by image links or ASCII diagrams in a GitHub repo.

---

# Automatic Differentiation (AD)

The first time that Professor Edelman heard about automatic differentiation (AD), it was easy for him to imagine what it was... but what he imagined was wrong! In his head, he thought it was straightforward symbolic differentiation applied to code—sort of like executing Mathematica or Maple, or even just automatically doing what he learned to do in his calculus class.

For instance, just plugging in functions and their domains from something like the following first-year calculus table:

| Derivative                    | Domain |
|-----------------------------|--------|
| $(\sin x)' = \cos x$          | $- \infty< x < \infty$ |
| $(\cos x)' = - \sin x$        | $-\infty< x < \infty$ |
| $(\tan x)' = \sec^2 x$        | $x\neq \frac{\pi}{2} + \pi n, n \in \mathbb{Z}$ |
| $(\cot x)' = - \csc^2 x$      | $x\neq \pi n, n \in \mathbb{Z}$ |
| $(\sec x)' = \tan x \sec x$   | $x \neq \frac{\pi}{2} + \pi n, n\in \mathbb{Z}$ |
| $(\csc x)' = - \cot x \csc x$ | $x\neq \pi n, n \in \mathbb{Z}$ |

And in any case, if it wasn't just like executing Mathematica or Maple, then it must be finite differences, like one learns in a numerical computing class.

It turns out that it is definitely *not* finite differences—AD algorithms are generally exact (in exact arithmetic, neglecting roundoff errors), not approximate. But it also doesn't look much like conventional symbolic algebra: the computer doesn't really construct a big “unrolled” symbolic expression and then differentiate it.

## Automatic Differentiation via Dual Numbers

### Example: Babylonian Square Root

We consider the “Babylonian” algorithm to compute $\sqrt{x}$, known for millennia (and a special case of Newton's method):

```julia
function Babylonian(x; N = 10) 
    t = (1+x)/2   # one step from t=1
    for i = 2:N   # remaining N-1 steps
        t = (t + x/t) / 2
    end    
    return t
end
```

Running the function on `x=4`:

```julia
julia> Babylonian(4, N=1)
2.5

julia> Babylonian(4, N=2)
2.05

julia> Babylonian(4, N=3)
2.000609756097561

julia> Babylonian(4, N=4)
2.0000000929222947

julia> Babylonian(4, N=10)
2.0
```

To compute derivatives automatically, we define a new number type `D`:

```julia
struct D <: Number
    val::Float64
    deriv::Float64
end

Base.:+(x::D, y::D) = D(x.val + y.val, x.deriv + y.deriv)
Base.:/(x::D, y::D) = D(x.val/y.val, (y.val*x.deriv - x.val*y.deriv)/y.val^2)

Base.convert(::Type{D}, r::Real) = D(r, 0)
Base.promote_rule(::Type{D}, ::Type{<:Real}) = D
```

Try it out:

```julia
julia> Babylonian(D(49, 1))
D(7.0, 0.07142857142857142)

julia> (√49, 0.5/√49)
(7.0, 0.07142857142857142)
```

### Dual Numbers

Dual numbers are defined as $a + b\epsilon$ with $\epsilon^2 = 0$. This leads to:

```julia
Base.:-(x::D, y::D) = D(x.val - y.val, x.deriv - y.deriv)
Base.:*(x::D, y::D) = D(x.val * y.val, x.deriv * y.val + x.val * y.deriv)
Base.show(io::IO, x::D) = print(io, x.val, " + ", x.deriv, "ϵ")
```

Example:

```julia
julia> ϵ = D(0,1)
0.0 + 1.0ϵ

julia> ϵ * ϵ 
0.0 + 0.0ϵ

julia> Babylonian(49 + ϵ)
7.0 + 0.07142857142857142ϵ
```

## Naive Symbolic Differentiation

This approach unrolls the entire expression before differentiating. Example:

- After 1 iteration: $(x+1)/2$
- After 2: $((x+1)/2 + 2x/(x+1))/2 = \frac{x^2 + 6x + 1}{4(x+1)}$

It quickly becomes computationally infeasible as expression size grows exponentially.

## Automatic Differentiation via Computational Graphs

Given:

```
a(x,y) = sin(x)
b(x,y) = a(x,y)/y
z(x,y) = b(x,y) + x
```

We can draw a computational graph and use chain rule on it to find:

```
∂z/∂x = (cos x)/y + 1
∂z/∂y = -sin x / y^2
```

Each node stores `(value, derivative)`. You propagate using:

```
(value, path product) ⟶ (f(value), f' * path product)
```

## Reverse Mode Automatic Differentiation

In reverse mode, you move from outputs to inputs. At each node:

```
∂z/∂a = Σ ∂b_i/∂a * ∂z/∂b_i
```

Reverse mode is more efficient than forward mode for functions where the number of inputs $n \gg m$ (e.g., neural networks).

## Forward vs. Reverse Mode Summary

| Mode         | Cost Scales With | Best For            |
|--------------|------------------|---------------------|
| Forward Mode | Number of inputs | Few inputs, many outputs |
| Reverse Mode | Number of outputs| Many inputs, few outputs |

Reverse mode requires storing a computation graph and traversing it backwards.

## Forward-over-Reverse Mode: Second Derivatives

- Compute gradient via reverse mode
- Compute Hessian or Hessian-vector product via forward mode
- Avoids forming full $n \times n$ matrix

### Example: Hessian-vector product

```julia
using ForwardDiff, Zygote, LinearAlgebra

f(x) = 1/norm(x)
g(z) = sum(z)^3
h(x) = g(Zygote.gradient(f, x)[1])

function ∇h(x)
    ∇f(y) = Zygote.gradient(f, y)[1]
    ∇g = Zygote.gradient(g, ∇f(x))[1]
    return ForwardDiff.derivative(α -> ∇f(x + α * ∇g), 0)
end

x = randn(5)
δx = randn(5) * 1e-8

h(x)
∇h(x)
∇h(x)' * δx
h(x + δx) - h(x)
```

### Exercise

Let `f(x, p) ∈ ℝ` and `g(z): ℝⁿ → ℝ`. Let `h(x,p) = g(∇_x f(x,p))`. Show:

```math
∇_p h(x, p) = \left.\frac{\partial}{\partial\alpha} \nabla_p f(x + \alpha ∇g(z), p) \right|_{\alpha = 0}
```

Try implementing in Julia!

---

Let me know if you'd like all the TikZ diagrams turned into SVG or PNG images for GitHub rendering.