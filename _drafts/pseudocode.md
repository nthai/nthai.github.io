---
layout: post
title: pseudocode example
---

{% include pseudocode.html id="1" code="
\begin{algorithm}
\caption{PPO with GAE for the continuing case}
\begin{algorithmic}
\PROCEDURE{Train}{$s, r, a, \pi_a, s'$} \COMMENT{the inputs are tensors of batches}
    \STATE $\hat{r} \leftarrow (1 - \alpha^{R})\hat{r} + \alpha^R \text{mean}(r)$
    \FOR{$k$ epochs}
        \STATE $\delta \leftarrow r - \hat{r} + V(s') - V(s)$
        \STATE $A\leftarrow$ compute GAE advantages from $\delta$ with given $\lambda$.
    \ENDFOR
\ENDPROCEDURE
\PROCEDURE{Partition}{$A, p, r$}
    \STATE $x = A[r]$
    \STATE $i = p - 1$
    \FOR{$j = p$ \TO $r - 1$}
        \IF{$A[j] < x$}
            \STATE $i = i + 1$
            \STATE exchange
            $A[i]$ with     $A[j]$
        \ENDIF
        \STATE exchange $A[i]$ with $A[r]$
    \ENDFOR
\ENDPROCEDURE
\end{algorithmic}
\end{algorithm}
" %}