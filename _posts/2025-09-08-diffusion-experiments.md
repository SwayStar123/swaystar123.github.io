---
title: 'My Diffusion Experiments'
date: 2025-09-08
permalink: /posts/2025/09/diffusion-experiments/
tags:
  - diffusion
  - ai
---

Here I note down some experiments I tried with diffusion models in the past or that I plan on conducting (if i get the compute for it)


## Improving SRA

[SRA](https://vvvvvjdy.github.io/sra) is a paper which introduces a method of self alignment, essentially, aligns an earlier layer at a higher timestep with a later layer with a lower timestep. I really like this idea, as it allows the paper to compete with (but unforunately not overtake) REPA, *without involving an externally trained model*. Theoretically it could outperform REPA too if you trained a large enough model with this method, as they show that the benefits from SRA increase as the model size gets larger (but they only test till XL).

So, if this method could be improved further, enough to overtake REPA, it will probably gain alot more recognition.

### E-SRA

What I called Entangled-SRA was inspired by [REG](https://github.com/Martinser/REG), which is a paper which improves REPA, it trains the diffusion model with repa, but additionally also includes the dino cls token as an additional target token to be denoised from scratch, which allows it to learn alot faster, 23x faster than REPA infact. 

So the idea was to do SRA, but then also pool the teacher layers activations to form a cls token, and then denoise that aswell. However unforunately this didnt work out that well.

I also implemented [CFM](https://github.com/gstoica27/DeltaFM) which should give another 3x boost

Results:

```
SiT-XL/2: (CFM, E-SRA). CLS weight 0.03
    300k train steps:
        Inception Score: 81.48990631103516
        FID: 14.728037552378794
        sFID: 8.32263929140015
        Precision: 0.65464
        Recall: 0.6173
```
The results were actually nearly indentical to SRA at 300k steps, so this was a null result. It is unclear if the CFM was counteracting any negative effect from my method, or the other way around. I also tried increasing the CLS weight to 1

```
SiT-B/2: (CFM, E-SRA). CLS weight 1.0
    375k train steps:
        Inception Score: 36.352333068847656
        FID: 36.51545321657193
        sFID: 9.45342136689294
        Precision: 0.49322
        Recall: 0.6285
```
however this was actually significantly worse than the SRA's FID.

In retrospect I should have tested just CFM + SRA first in isolation, and my method in isolation! With these unknown confounding factors it is hard to determine what went wrong here. I made the mistake of trying to do multiple things together as a yolo run because I was on a limited compute budget.

However I theorize that my method might not work that well as it is. The main benefit from SRA is that an earlier layer is forced to model a later layer, which aligns itself to work more efficiently, however in my case when I am diffusing the pooled embedding, that happens after all the layers, so the teacher layer is actually shallower than the prediction head.

If I have additional compute and the desire to continue in this direction I will probably have to add a velocity head at the student layer and get the denoising prediction there, rather than at the end.

### Modifying teacher timestep distribution

In the SRA paper, they ablate over their teachers timestep distribution, but they only test distributions in which the teacher timestep is 0-0.3 ahead of the student. Which I find a bit too low (and the fact that it is a random distribution irritates me! Isnt this a inconsistent target? The representations at t and t-0.3 could be very different! And the student has no idea which of this it is supposed to model!)

This is their current teacher timestep calculation method: 
```python
time_input_teacher = time_input - (self.t_max * torch.rand_like(time_input))
```
where t_max is by default 0.2

I replace it with a simple

```python
time_input_teacher = time_input / 10
```
which makes it so at t=1, teacher time = 0.1, and as student time approaches 0, teacher time also approaches 0.

```
Inception Score: 48.172080993652344
FID: 30.437015441628148
sFID: 6.288378112108376
Precision: 0.5589
Recall: 0.6347
```

SRA with the default settings on B/2 gets 29.10 FID at 400k steps, so this is slightly worse, but curiously, my results are still better than their constant time interval of 0.2 ablation (which gets 30.7). So if I do time_input / 5 or something instead it might be better?

### TODOs

SRA + [SARA](https://arxiv.org/abs/2503.08253) might work, or a modified version of SARA. Or maybe just distilling the attention queries and keys at the teacher layer into the student layer is also enough.

[Dispersive loss](https://arxiv.org/abs/2506.09027) along with SRA could be good. SRA shows that the better the teacher model, the more useful SRA is, so the benefit from dispersive loss could compound.

[Contrastive flow loss](https://github.com/gstoica27/DeltaFM) needs to be tested in isolation to see if it also stacks with SRA and delivers compounding gains.

[REPA](https://github.com/sihyun-yu/REPA) is the original which inspired SRA, but they dont test if combining the two work, further derivatives like REG/ReDI/REPA-E can also be tested.


## Reproducing Contrastive Flow Matching

[CFM](https://github.com/gstoica27/DeltaFM) is a paper which suggests a really simple way to get ~3x faster convergence (and faster sampling) with an additional auxillary loss. Unforunately they dont provide code, so I reproduced it here: https://github.com/SwayStar123/REPA

And successfully reproduced the numbers from the paper:

```
B/2 λ=0.05: 
  Inception Score: 69.62489318847656
  FID: 20.539321634715975
  sFID: 5.430992245223706
  regularized FD-DINOv2_eff: 1818.6091087211607
```

(Reported FID in paper is 20.5, 400k steps w REPA, B/2)

It is actually very simple to implement it! These are the few lines you need to add/change

[Loss calculation](https://github.com/SwayStar123/REPA/blob/5706326b92aeaa2ff5cc65bcf4705ec0516be1bd/loss.py#L90):
```python
        contrastive_flow_target = torch.roll(model_target, shifts=1, dims=0)
        contrastive_flow_loss = mean_flat((model_output - contrastive_flow_target) ** 2)
```
[Loss weight](https://github.com/SwayStar123/REPA/blob/5706326b92aeaa2ff5cc65bcf4705ec0516be1bd/train.py#L348):
```python
loss = (loss_mean - contrastive_flow_loss_mean * args.contrastive_flow_coeff) + proj_loss_mean * args.proj_coeff
```
(Projection loss is the REPA loss)

### Improving CFM

I tried improving CFM further by introducing a time weighting to the loss coefficient. My thinking was that at lower timesteps (low noise), the clear image is very obvious to the model, so adding a contrastive flow loss to that training example could actually unnecessarily perterb the flow and hinder training. So I modified the loss calculation slightly to
```python
contrastive_flow_loss = mean_flat((model_output - contrastive_flow_target) ** 2) * self.contrastive_flow_schedule(time_input)
```
where the constrative flow schedule would be by default a linear schedule:
```python
def linear_schedule(t):
    return t
```

I planned on testing other schedules too if this was successful, but unforunately:

```
B/2 linear schedule λ=0.05:
  Inception Score: 65.89761352539062
  FID: 23.426444638420435
  sFID: 6.770656642394329
  regularized FD-DINOv2_eff: 1792.0649000826197
```
The results were worse! I thought maybe because the average t is 0.5, the loss weighting is getting halved, so if we double the lambda to counteract it, then it could start becoming better?

```
B/2 linear schedule λ=0.1:
  Inception Score: 67.88655090332031
  FID: 21.47094195726106
  sFID: 5.468999414535688
  regularized FD-DINOv2_eff: 1813.040922549716
```

It got better, but unforunately still underperforms the original CFM implementation. Due to these disappointing results I did not test the other schedules or higher lambda values.