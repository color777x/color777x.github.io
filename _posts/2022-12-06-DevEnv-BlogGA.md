---
title: "MathJax LaTex 사용법"
header:
categories:
  - Development Environment
tags:
  - MathJax
  - Blog
use_math: true
---



# LaTex 수식 사용하기

* [참고 블로그 1](http://egloos.zum.com/scienart/v/2665978), [참고 블로그 2](https://khw11044.github.io/blog/blog-etc/2020-12-21-markdown-tutorial2/)

우선 수식 앞 뒤로 '$'표시를 사용한다.

```latex
$1+1=2$, $1-1=0$
```

- 곱셈

```latex
$3 \times 3 = 9$
```

$$
3 \times 3 = 9
$$



- 나눗셈

```latex
$10 \div 2 = 5$
```

$$
10 \div 2 = 5
$$



* 분수

```latex
$\frac{3}{5})$
```

$$
\frac{3}{5}
$$

* 위 첨자

```latex
$2^2 = 4$
```

$$
2^2 = 4
$$

* 아래 첨자

```latex
$K_1$
```

$$
K_1
$$

* 루트

```latex
$\sqrt{2}$
```

$$
\sqrt{2}
$$

* 펙토리얼

```latex
$n!$
```

$$
n!
$$

```latex
$n! = \prod_{k=1}^n k$
```

$$
n! = \prod_{k=1}^n k
$$

* 합집합 교집합 차집합

```latex
${A} \cup {B}$, ${A} \cap {B}$, ${A} \in {B}$
```

$$
{A} \cup {B} , {A} \cap {B}, {A} \in {B}
$$

* 사각함수

```latex
$sin\theta = \frac{y}{r}$
```

$$
sin\theta = \frac{y}{r}
$$

* 파이

```latex
$\pi$, $\Pi$, $\phi$
```

$$
\pi, \Pi, \phi
$$

* 각도

```latex
$90^\circ$
```

$$
90^\circ
$$

* 극한

```latex
$\lim_{x \to \infty} \exp(-x) = 0$
```

$$
\lim_{x \to \infty} \exp(-x) = 0
$$

```latex
$\frac{df(x)}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}$
```

$$
\frac{df(x)}{dx} = \lim_{h \to 0} \frac{f(x+h) - f(x)}{h}
$$

* 시그마

```latex
$$\sum_{i=1}^{10} t_i$$  
```

$$
\sum_{i=1}^{10} t_i \\
$$

$$
MSE = \frac{1}{N}\sum_{i=1}^{N}(y_1 -   \hat{y})^2
$$

* log

```latex
$CE = -\sum_{i}^{C}t_i \log(f(s)_i)$
```


$$
CE = -\sum_{i}^{C}t_i \log(f(s)_i)
$$

* 미분

```latex
$\dv{Q}{t} = \dv{s}{t}$
```

$$
\dv{Q}{t} = \dv{s}{t}
$$

* 적분

```latex
$$\int_0^\infty \mathrm{e}^{-x}\,\mathrm{d}x$$

$$\int\limits_a^b$$

$$ (f*g)(t) = \int_{-\infty}^{\infty }f(τ)g(t-τ)dτ $$
```

$$
\int_0^\infty \mathrm{e}^{-x}\,\mathrm{d}x, \int\limits_a^b \\

(f*g)(t) = \int_{-\infty}^{\infty }f(τ)g(t-τ)dτ
$$





### 논문에 자주 나오는 기호 [Permalink](https://khw11044.github.io/blog/blog-etc/2020-12-21-markdown-tutorial2/#%EB%85%BC%EB%AC%B8%EC%97%90-%EC%9E%90%EC%A3%BC-%EB%82%98%EC%98%A4%EB%8A%94-%EA%B8%B0%ED%98%B8)

**특수문자**

| 이름   | 명령어      | 반환   |      | 이름   | 명령어      | 반환   |
| ---- | -------- | ---- | ---- | ---- | -------- | ---- |
| 알파   | \alpha   | α    |      | 크사이  | \xi      | ξ    |
| 베타   | \beta    | β    |      | 오미크론 | o        | o    |
| 감마   | \gamma   | γ    |      | 파이   | \pi      | π    |
| 델타   | \delta   | δ    |      | 로    | \rho     | ρ    |
| 엡실론  | \epsilon | ϵ    |      | 시그마  | \sigma   | σ    |
| 제타   | \zeta    | ζ    |      | 타우   | \tau     | τ    |
| 에타   | \eta     | η    |      | 입실론  | \upsilon | υ    |
| 세타   | \theta   | θ    |      | 파이   | \phi     | ϕ    |
| 이오타  | \iota    | ι    |      | 카이   | \chi     | χ    |
| 카파   | \kappa   | κ    |      | 오메가  | \omega   | ω    |
| 람다   | \lambda  | λ    |      | 뉴    | \nu      | ν    |
| 뮤    | \mu      | μ    |      |      |          |      |

**관계연산자**

| 이름     | 명령어     | 반환   |      | 이름     | 명령어     | 반환   |
| ------ | ------- | ---- | ---- | ------ | ------- | ---- |
| 합동     | \equiv  | ≡    |      | 근사     | \approx | ≈    |
| 비례     | \propto | ∝    |      | 같고 근사  | \simeq  | ≃    |
| 닮음     | \sim    | ∼    |      | 같지 않음  | \neq    | ≠    |
| 작거나 같음 | \leq    | ≤    |      | 크거나 같음 | \geq    | ≥    |
| 매우작음   | \ll     | ≪    |      | 매우 큼   | \gg     | ≫    |

**논리기호**

| 이름           | 명령어             | 반환   |      | 이름            | 명령어         | 반환   |
| ------------ | --------------- | ---- | ---- | ------------- | ----------- | ---- |
| 불릿           | \bullet         | ∙    |      | 부정            | \neq        | ≠    |
| wedge        | \wedge          | ∧    |      | vee           | \vee        | ∨    |
| 논리합          | \oplus          | ⊕    |      | 어떤            | \exists     | ∃    |
| 오른쪽 </br>화살표 | \rightarrow     | →    |      | 왼쪽 <\br>화살표   | \leftarrow  | ←    |
| 왼쪽 <\br>큰화살표 | \Leftarrow      | ⇐    |      | 오른쪽 <\br>큰화살표 | \Rightarrow | ⇒    |
| 양쪽 <\br>큰화살표 | \Leftrightarrow | ⇔    |      | 양쪽 <\br>화살표   | \leftarrow  | ←    |
| 모든           | \forall         | ∀    |      |               |             |      |

**집합기호**

| 이름     | 명령어         | 반환   |      | 이름    | 명령어                    | 반환   |
| ------ | ----------- | ---- | ---- | ----- | ---------------------- | ---- |
| 교집합    | \cap        | ∩    |      | 합집합   | \cup                   | ∪    |
| 상위집합   | \supset     | ⊃    |      | 진상위집합 | \supseteq              | ⊇    |
| 하위집합   | \subset     | ⊂    |      | 진하위집  | \subseteq              | ⊆    |
| 부분집합아님 | \not\subset | ⊄    |      | 공집합   | \emptyset, \varnothing | ∅ ∅  |
| 원소     | \in         | ∈    |      | 원소아님  | \notin                 | ∉    |

**기타**

| 이름     | 명령어         | 반환   |      | 이름       | 명령어           | 반환   |
| ------ | ----------- | ---- | ---- | -------- | ------------- | ---- |
| hat    | \hat{x}     | x^   |      | widehat  | \widehat{x}   | x^   |
| 물결     | \tilde{x}   | x~   |      | wide물결   | \widetilde{x} | x~   |
| bar    | \bar{x}     | x¯   |      | overline | \overline{x}  | x¯   |
| check  | \check{x}   | xˇ   |      | acute    | \acute{x}     | x´   |
| grave  | \grave{x}   | x`   |      | dot      | \dot{x}       | x˙   |
| ddot   | \ddot{x}    | x¨   |      | breve    | \breve{x}     | x˘   |
| vec    | \vec{x}     | x→   |      | 델,나블라    | \nabla        | ∇    |
| 수직     | \perp       | ⊥    |      | 평행       | \parallel     | ∥    |
| 부분집합아님 | \not\subset | ⊄    |      | 공집합      | \emptyset     | ∅    |
| 가운데 점  | \cdot       | ⋅    |      | …        | \dots         | …    |
| 가운데 점들 | \cdots      | ⋯    |      | 세로점들     | \vdots        | ⋮    |
| 나누기    | \div        | ÷    |      | 물결표      | \sim          | ∼    |
| 플마,마플  | \pm, \mp    | ± ∓  |      | 겹물결표     | \approx       | ≈    |
| prime  | \prime      | ′    |      | 무한대      | \infty        | ∞    |
| 적분     | \int        | ∫    |      | 편미분      | \partial      | ∂    |
| 한칸띄어   | x \, y      | xy   |      | 두칸       | x\;y          | xy   |
| 네칸띄어   | x \quad y   | xy   |      | 여덟칸띄어    | x \qquad y    |      |

































