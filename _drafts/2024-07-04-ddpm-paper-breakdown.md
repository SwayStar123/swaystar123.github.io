---
title: 'Denoising Diffusion Probabilistic Models - Paper breakdown'
date: 2024-07-04
permalink: /posts/2024/07/ddpm-paper-breakdown/
tags:
  - diffusion
  - ai
---

- [Overview](#overview)
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Background](#background)
- [Diffusion models and denoising autoencoders](#diffusion-models-and-denoising-autoencoders)
  - [Forward process and L\_T](#forward-process-and-l_t)
  - [Reverse process and L\_T:T-1](#reverse-process-and-l_tt-1)
  - [Data scaling, reverse process decoder, and L\_0](#data-scaling-reverse-process-decoder-and-l_0)
  - [Simplified training objective](#simplified-training-objective)
- [Experiments](#experiments)
  - [Sample quality](#sample-quality)
  - [Reverse process paramertization and training objective ablation](#reverse-process-paramertization-and-training-objective-ablation)
  - [Progressive coding](#progressive-coding)
  - [Interpolation](#interpolation)
- [Related Work](#related-work)
- [Conclusion](#conclusion)

# Overview

Denoising Diffusion Probabilistic Models (DDPM for short) are the foundation for diffusion models. This blog post will breakdown the entire paper with commentary and additional images/videos in order to aide in understanding of all the maths and concepts.

Thank you to Jonathan Ho, Ajay Jain, and Pieter Abbeel from UC Berkeley for their work on diffusion models.

[Link to original paper](https://arxiv.org/abs/2006.11239)

# Abstract

> We present high quality image synthesis results using diffusion probabilistic models,
a class of latent variable models inspired by considerations from nonequilibrium
thermodynamics. Our best results are obtained by training on a weighted variational
bound designed according to a novel connection between diffusion probabilistic
models and denoising score matching with Langevin dynamics, and our models naturally admit a progressive lossy decompression scheme that can be interpreted as a
generalization of autoregressive decoding. On the unconditional CIFAR10 dataset,
we obtain an Inception score of 9.46 and a state-of-the-art FID score of 3.17. On
256x256 LSUN, we obtain sample quality similar to ProgressiveGAN. Our implementation is available at https://github.com/hojonathanho/diffusion.

# Introduction

# Background

# Diffusion models and denoising autoencoders

## Forward process and L_T  

## Reverse process and L_T:T-1

## Data scaling, reverse process decoder, and L_0

## Simplified training objective

# Experiments

## Sample quality

## Reverse process paramertization and training objective ablation

## Progressive coding

## Interpolation

# Related Work

# Conclusion