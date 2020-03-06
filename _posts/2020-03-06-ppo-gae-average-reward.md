---
layout: post
title: Using average reward for generalized advantage estimation in continuing problems
---

Most of the RL studies in the literature investigate episodic problems so it is a bit difficult to find algorithms and pytorch implementations on Github that handle continuing problems. In some simple cases we could simply use an episodic RL agent and just never feed it a terminal state. However when I tried to use a more complicated agent (eg.: the PPO agent from minimalRL) I bumped into convergence problems where the agent simply couldn't find a good policy. I'm still not sure what is the root cause of the problem, but rewriting the update method into an average reward setting fixed it for me.

According to Sutton (RL: An Intro, chapter 10.3) the discounted reward setting does not work with function approximation and in this case we need to replace it with the average reward setting. This is a very simple substitution, we just need to replace $$r_i$$ with $$r_i - \bar{r}$$ and remove every $$\gamma$$. For the simple case of an actor critic Sutton already provides an algorithm (RL: An Intro, chapter 13.6). I thought I would jot down the same with minor modifications and with Schulman's GAE.

First we will look at the TD-errors $$\delta_t$$, which are also an estimate of the advantage $$A(s_t, a_t)$$, in the average reward setting.
<p align="center"> $$\delta_t = r_t - \bar{r} + V(s_{t+1}) - V(s_t)$$ </p>

### Sources:
* [Mao et al.: *Park: An Open Platform for Learning-Augmented Computer Systems*](https://papers.nips.cc/paper/8519-park-an-open-platform-for-learning-augmented-computer-systems)
* [R. S. Sutton, A. G. Barto: *Reinforcement Learning: An Introduction*](http://incompleteideas.net/book/the-book-2nd.html)
* [Schulman et al.: *High-Dimensional Continuous Control Using Generalized Advantage Estimation*](https://arxiv.org/abs/1506.02438)
* [seungeunrho: *minimalRL*](https://github.com/seungeunrho/minimalRL)
