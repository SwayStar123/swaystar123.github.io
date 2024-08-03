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

Then, using the key and the query, you can compute how much "attention" a word pays to another word (or itself).
The below can be read as "the amount of attention token $$x_n$$ pays to token $$x_m$$".

$$ a[x_m, x_n] = softmax_m[\frac{q_m \cdot k_n^T}{\sqrt{D_k}}] $$

where $$D_k$$ is the dimension of the key and query vectors.

The numerator is the dot product of the query and the transpose of the key. This ends up being a scalar value since both the query and the key have the same dimension. This scalar value represents how similar the two vectors are, if they are similar, the value will be higher, and if not, the value will be lower. However, the values can become very large, as the dot product operation is a sum of products of the elements of the vectors, and as discussed above, ai training doesnt play well with big numbers, so to counteract this, we have the denominator.

The denominator is the squart root of the dimension of the vectors, this is to ensure training stability (in the softmax function, and if the values vary largely, the gradients will be unstable).

The softmax function is then applied to the term, which normalizes the value to between 0 and 1, where all the values along the m column sum to 1.
Essentially meaning, the values for all $$[x_(1, .., N), x_n]$$ sum to 1. So we have a list of which $$x_n$$s pay the most attention to the given $$x_m$$.

Going back to the example sentence "The black cat is sleeping", this could be a hypothetical attention matrix

$$x_n$$ | $$x_1$$ "The" | $$x_2$$ "black" | $$x_3$$ "cat" | $$x_4$$ "is" | $$x_5$$ "sleeping"
--- | --- | --- | --- | --- | ---
"The" | 0.15 | 0.05 | 0.8 | 0.0 | 0.0 
"black" | 0.0 | 0.1 | 0.9 | 0.0 | 0.0 
"cat" | 0.1 | 0.5 | 0.2 | 0.05 | 0.15 
"is" | 0.00 | 0.00 | 0.1 | 0.2 | 0.7 
"sleeping" | 0.1 | 0.0 | 0.8 | 0.05 | 0.05 

Heres my thought process behind the numbers i made up (its not necessary any model learns such a matrix, this is something hypothetical i made up to aide understanding)

"The": The word "cat" pays the most attention to this, because in this sentence, it marks the subject of the sentence.
"black": Again, the word "cat" pays most attention to this, because it is the cat that is black.
"cat": The words "black" and "sleeping" pay the most attention, however, the word also pays some attention to itself.
"is": The word "sleeping" pays the most attention, as that the action being done.
"sleeping": The word "cat" pays the most attention, as it is the one sleeping.

Using these attention weights, and the previously calculated value vectors, we will now begin to modify the token embeddings.

$$ sa_n[x_1, .., x_N] = \sum_{m=1}^{N} a[x_m, x_n] \cdot v_m $$

This is called self attention, as the keys, queries, and values, all come from the same sequence of token embeddings. (As opposed to cross attention, where not all the vectors are from the same sequence).

The attention scalars that we just calculated act as weights, that decide how much of each value vector affect the $$sa_n$$ output vector.

Continuing using the example sentence "The black cat is sleeping", and using the hypothetical attention matrix, here is an imagined explanation for what the values represent, and what they combine into for all the $$sa_n$$ vectors.

$$sa_1$$: "The" goes from an abstract embedding that contains the meaning of "subject marker", to a more concrete meaning of "The cat". The value matrix for $$m = 3$$ which is "cat" is the highest, which encodes the meaning of "cat" into the $$sa_1$$ vector.
$$sa_2$$: "black" goes from just the colour "black" to a more nuanced meaning, of "black (fur, breed of cat, etc)", as "cat" pays the most attention to this token.
$$sa_3$$: "cat" goes from an embedding representing the concept of the average cat, to a "The (singular subject) black sleeping cat", due the attention paid by all the other tokens.
$$sa_4$$: "is" goes from another abstract verb embedding, to a more nuanced embedding of "(animal) is (sleeping)" due to the attention paid by "cat" and "sleeping" .
$$sa_5$$: "sleeping" goes from a verb embedding more heavily weighted to humans (as thats the most common use of the word), to now representing cats sleeping, perhaps it has the added nuance of "purring" now too!

# Transformer Architecture

WIP. I dont have a good enough *intuitive* explanation for why the transformer architecture is as it is.