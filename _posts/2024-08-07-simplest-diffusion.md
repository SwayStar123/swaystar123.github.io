---
title: 'Simplest Diffusion'
date: 2024-08-07
permalink: /posts/2024/08/simplest-diffusion/
tags:
  - diffusion
  - ai
---

I present to the world, the simplest diffusion model!
Using only a fully connected feedforward network, and the dead simple sampling function of

```python
def sample(model, bs: int = 9, steps: int = 100):
    x = torch.randn(bs, 28*28).to(device)

    pred_weight = 4/steps
    x_weight = 1 - pred_weight

    for _ in range(steps):
        pred = model.forward(x)
        x = x * x_weight + pred * pred_weight
    return x
```

we can generate images like this from pure noise.

![generated samples](/images/blog_images/2024-08-07-simplest-diffusion/generated_samples.png)

The sampling function could probably be improved, as the images are not perfect (or maybe you need to train the model for longer, only trained for 10 epochs)

Full code can be seen [here](https://github.com/SwayStar123/simplest_diffusion/)

The code is so simple i dont even think it needs any explanation, id just be repeating myself.


