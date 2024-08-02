---
title: 'Transformers in Brief'
date: 2024-08-02
permalink: /posts/2024/08/transformers-in-brief/
tags:
  - transformers
  - ai
---

My attempt to intuitively explain transformers in brief.

# Overview

Transformers are a type of neural network architecture designed originally to handle text information.

# Tokens and the tokenizer.

## Problem
You have text data. Unforunately, machines only speak the language of numbers. You need to convert the text into numbers so the ai can understand it.

## Solution
You can make a dictionary for the ai, where you assign each word a number, which is its `word id` (also called `token id`).

Here is an example dictionary, containing a very small vocabulary, and a vocabulary size of 9.

Word | Token ID
--- | ---
black | 0
grey | 1
brown | 2
cat | 3
dog | 4
sleeping | 5
playing | 6
is | 7
the | 8

Using this dictionary you can convert your text into a sequence of numbers. For example, with the sentence "The black cat is sleeping" you would get the sequence `[8, 0, 3, 7, 5]`. (Ignoring the spaces and capitalization).

## Problem 2
In the given example, the vocabulary size is only 9, in the real world, you would want your ai to know as many words as possible. You could just scale this up to millions, however there would be a problem. The ai would be unable to deal with made up words, words not common enough to make it into the dictionary, names not common enough to make it into the dictionary, etc.

## Solution
You use a tokenizer!

A tokenizer analyzes a huge corpus of text, and learns what strings are the most frequent. It greedily tries to use the least amount of tokens to represent the entire text dataset, given a constraint on the vocabulary size. This means that in the most advanced LLM tokenizers, there is most likely a token for every character, space, punctuation mark, etc, on top of the tokens for most of the words. Not only would the tokenizer learn the most common words, given a large enough vocabulary size, it would also pick up on the most common colocation of words, for example the phrase "Hello world!" is extremely common, and could potentially be represented by a single token. Whereas for an uncommon name thats very rarely used, the tokenizer would likely have to make it up using multiple tokens.

You can see how the openai tokenizers split up a piece of text here https://platform.openai.com/tokenizer
Curiously, the openai tokenizer usually has tokens for "{space}+{word}" instead of just "{word}", so it actually rarely uses the space token. This makes sense if you think about it though, as the tokenizer is far more likely to see words with spaces seperating them, so it just learns the tokens along with the space.

# Embeddings

## Problem
You converted all your text into a sequence of numbers, but these numbers are still meaningless to the ai. You need to convert these token/word ids into some numbers that the ai can actually understand!

## Solution
You use embeddings!

You can think of embeddings as yet another dictionary, this time mapping the token ids to a vector of numbers. All of these vectors would be *learnable parameters*. What this means is that instead of them being chosen by humans or an algorithm, the AI will decide these vectors for itself during training. This bridges the gap between numbers we can understand, and the numbers the ai can understand.

# Positional Encoding

## Problem
Further ahead, we will want every token to interact with every other token. However, in that situation, the order of words will be lost. "he killed the lion" and "the lion killed him" are two different sentences, in the embedding form, they would retain their positions, but when figuring out their interactions, you would consider every possible pair of tokens, in which the order of the tokens would be lost.

## Solution
You use positional encoding!

Positional encodings are just a bunch of sine waves, of different frequencies, so if you sample the positional embedding, you will get different values for each position.

The reason you dont just use a simple range of increasing numbers (ie, [0, 1, 2, 3, 4, ...]) is because ais generally dont play well with big numbers, and the sequence length could go upto millions in the most advanced models. So to keep the numbers between -1 and 1, you use sine waves.

As the positional encoding is constant, it is just precomputed for the entire range of sequence lengths that the ai will see during training, and just looked up when needed.

The positional encodings are added to the token embeddings, and the result is fed into the attention layer.

# Attention

## Problem
You have a sequence of token embeddings, but the length of the sequence can vary. You want to be able to support sentences of all sizes, paragraphs, essays, books, etc. In order to support all these extremely varying sequence lengths, there is no obvious way to use a fully connected neural network.

## Solution
The attention architecture solves this problem by using the same set of weight matrices for every token embedding in the sequence, within the same attention layer (different layers have different weight matrices, and multi headed attention layers can have multiples of the weight matrices, but the number of these matrices wont change depending on the sequence length).

In a single single headed attention layer you have 3 weight matrices, $$W_q$$, $$W_k$$, and $$W_v$$, each representing a query, key, and value matrix respectively.

You obtain the query, key, and value vectors for each token embedding by multiplying the token embedding by the query, key, and value matrices respectively.

$$q_i = W_q \cdot x_i$$
$$k_i = W_k \cdot x_i$$
$$v_i = W_v \cdot x_i$$

where $$x_i$$ is the token embedding for the $$i$$th token in the sequence.

# Transformer Architecture

