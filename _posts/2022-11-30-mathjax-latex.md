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

* [참고 블로그](http://egloos.zum.com/scienart/v/2665978)

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

$$\displaystyle\sum_{i=1}^{10} t_i$$
```

$$
\sum_{i=1}^{10} t_i \\

\displaystyle \sum_{i=1}^{10} t_i
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




