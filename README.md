# Adversarial Detection Attack on AI-Generated Text

This repository contains a framework for performing adversarial attacks on AI-generated text detection models by leveraging tokens with high gradients. The approach is based on the [paper](https://arxiv.org/abs/2404.01907) "Humanizing Machine-Generated Content: Evading AI-Text Detection through Adversarial Attack" by Ying Zhou, Ben He, and Le Sun.

## Overview

The goal of this framework is to perturb AI-generated text in a way that makes it difficult for detection models to accurately classify it as machine-generated. This is achieved by identifying tokens with high importance using gradients, and then replacing these tokens with suitable alternatives using a masked language model (MLM).

## Features

- **Gradient-Based Token Importance**: Compute gradients to identify the most important tokens in the input text.
- **Masked Language Model (MLM)**: Use an MLM to generate synonym candidates for important tokens.
- **Adversarial Text Generation**: Replace important tokens with generated synonyms to create adversarial examples that evade detection.

## How to Use

You can load the CoLab notebook, or follow the installation guide to run locally. A base CoLab runtime will run it fine without using a GPU.

## Installation and Running

`pip install -r requirements.txt`

Be sure to set your GPT Zero API key in the `.env` file as `gptzero_api_key`, or you can remove this module and replace it with your own detector method.

Add your AI generated text or use the current placeholder text.

## Future or Additions

- Enhancing the replacement using NLP methods to ensure content and context remain the same.
- Implement a method to calculate the importance of each word in the input text using both gradient norms and perplexity changes.
- Improve the generate_adversarial_example function to use an MLM to generate multiple candidate replacements for each important word.

## Contact

Reach me at will@relativecompanies.com

## Reference

```
@misc{zhou2024humanizing,
      title={Humanizing Machine-Generated Content: Evading AI-Text Detection through Adversarial Attack}, 
      author={Ying Zhou and Ben He and Le Sun},
      year={2024},
      eprint={2404.01907},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
