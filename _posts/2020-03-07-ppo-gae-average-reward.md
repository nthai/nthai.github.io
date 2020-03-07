---
layout: post
title: Using average reward for generalized advantage estimation in continuing problems
---

Most of the RL studies in the literature investigate episodic problems so it is a bit difficult to find algorithms and pytorch implementations on Github that handle continuing problems. Recently Mao et al. released the Park paper in which they address such problems, it has a really good summary on computer system related problems, I really encourage everyone to read it. In some simple cases we could simply use an episodic RL agent and just never feed it a terminal state. However when I tried to use a more complicated agent (eg.: the PPO agent from minimalRL) I bumped into convergence problems where the agent simply couldn't find a good policy. I'm still not sure what is the root cause of the problem, but rewriting the update method into an average reward setting fixed it for me.

According to Sutton (RL: An Intro, chapter 10.3) the discounted reward setting does not work with function approximation and in this case we need to replace it with the average reward setting. This is a very simple substitution, we just need to replace $$r_i$$ with $$r_i - \bar{r}$$ and remove every $$\gamma$$ (for example by assuming $$\gamma = 1$$). For the simple case of an actor critic Sutton already provides an algorithm (RL: An Intro, chapter 13.6). I thought I would jot down the same with minor modifications and with Schulman's GAE.

First we will look at the TD-errors $$\delta_t$$, which are also an estimate of the advantage $$A(s_t, a_t)$$, in the average reward setting.
<p align="center"> $$\delta_t = r_t - \bar{r} + V(s_{t+1}) - V(s_t)$$ </p>

In the discounted reward setting $$\hat{A}^{(k)}_{t}$$ is the discounted sum of $$k$$ consecutive $$\delta$$ TD-errors. In the average reward setting we weill get the folowing.
<p align="center">
    $$\begin{array}{lll}
        \hat{A}^{(1)}_{t} &= \delta_{t}                   &= r_t  - \bar{r} + V(s_{t+1}) - V(s_t) \\
        \hat{A}^{(2)}_{t} &= \delta_{t} + \delta_{t+1}    &= r_t  - \bar{r} + r_{t+1} - \bar{r} + V(s_{t+2}) - V(s_t) \\
        \hat{A}^{(k)}_{t} &= \sum_{i=0}^{k-1}\delta_{t+i} &= \sum_{i=0}^{k-1}r_{t+i}  - k\bar{r} + V(s_{t+k}) - V(s_t)
    \end{array}$$
</p>

Then the generalized advantage estimator will become:
<p align="center">
    $$\hat{A}^{GAE}_{t} = (1- \lambda) (\hat{A}^{(1)}_t + \lambda \hat{A}^{(2)}_{t} + \lambda^2\hat{A}^{(3)}_{t} + \ldots) = \sum_{i=0}^{\infty}\lambda^i \delta_{t+i}. $$
</p>
Noe we can use this estimator and plug it into an actor-critic or PPO algorithm. Below is an example pseudocode, I tried to follow the notation used by Sutton.

{% include pseudocode.html id="1" code="
\begin{algorithm}
\caption{PPO with GAE for the continuing case}
\begin{algorithmic}

    \FOR{each timestep}
        \STATE $a \sim \pi_{\theta}( \cdot \vert s)$
        \STATE Take action $a$ and observe $s'$ and $r$.
        \STATE $\bar{r} \leftarrow (1-\alpha^R)\bar{r} + \alpha^R r$
        \FOR{$k$ epochs}
            \STATE $\text{TD}_{\text{target}} \leftarrow r - \bar{r} + V_{w}(s')$
            \STATE $\delta \leftarrow \text{TD}_\text{target} - V_{w}(s)$
            \STATE $\hat{A}^{GAE} \leftarrow$ Compute from $\delta$ using the equation with given $\lambda$.
            \STATE $w \leftarrow w + \alpha^{w}\text{TD}_\text{target}\nabla_w V_{w}(s)$
            \STATE $r_t(\theta) \leftarrow \frac{\pi_{\theta}(a \vert s)}{\pi_{\theta_{\text{old}}(a \vert s)}}$
            \STATE $\theta \leftarrow \theta + \alpha^{\theta}\nabla_\theta \min\{r_{t}(\theta)\hat{A}^{GAE}, \text{clip}(r_t(\theta), 1-\varepsilon, 1+\varepsilon)\hat{A}^{GAE}\}$
        \ENDFOR
        \STATE $s \leftarrow s'$
    \ENDFOR

\end{algorithmic}
\end{algorithm}
" %}


Below I'm also including a pytorch implementation of the above algorithm.
{% gist 38ed9c5fff6f4cadadf3d97152942407 %}

### Sources:
* [Mao et al.: *Park: An Open Platform for Learning-Augmented Computer Systems*](https://papers.nips.cc/paper/8519-park-an-open-platform-for-learning-augmented-computer-systems)
* [R. S. Sutton, A. G. Barto: *Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html)
* [Schulman et al.: *High-Dimensional Continuous Control Using Generalized Advantage Estimation*](https://arxiv.org/abs/1506.02438)
* [seungeunrho: *minimalRL*](https://github.com/seungeunrho/minimalRL)
