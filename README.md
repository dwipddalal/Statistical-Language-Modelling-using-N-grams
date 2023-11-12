# Statistical Language Modeling Using N-Grams

This project implements a statistical language model using N-grams with Laplace smoothing built on the Wikipedia corpus. It includes an interactive interface for displaying word predictions based on the N-gram model.

## Overview

This repository contains the Python implementation (`Code.py`) of an N-gram language model, which is a type of probabilistic language model used for predicting the next item in a sequence. The code includes preprocessing steps, model training with Laplace smoothing, and a method for calculating the perplexity of the model. Additionally, the Jupyter notebook (`Stastical_Language_Modelling_Using_N_grams.ipynb`) provides an interactive exploration of the model's capabilities.

## Features

- Implementation of N-gram language models (Unigram, Bigram, Trigram, etc.).
- Preprocessing of text data with special tokens for sentence boundaries and unknown words.
- Replacement of singletons (words occurring only once) with a special `<UNK>` token to handle out-of-vocabulary words.
- Laplace smoothing to account for unseen N-grams in the training data.
- Perplexity calculation to evaluate the model's performance.
- Prediction of the next word given a sequence of words.
- Serialization of the trained model for later use or deployment.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.x
- NLTK library (Natural Language Toolkit)

## Installation

Clone the repository to your local machine:

```shell
git clone https://github.com/dwipddalal/Statistical-Language-Modelling-using-N-grams.git
