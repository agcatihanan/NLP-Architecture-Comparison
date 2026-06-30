# NLP Architecture Comparison: BERT, GPT, and Text-GAN
**Authors:** Catihanan, Nase, Reyes  
**Program:** School of Information Technology 

## Project Overview
This repository contains the structured data processing, model pipelines, training histories, and test inferences for implementing and comparing three distinct NLP architectures: a discriminative model (BERT), an autoregressive generative model (GPT), and an adversarial generative model (Text-GAN).

**Dataset Repository Link:** [Stanford NLP IMDB Movie Reviews (Hugging Face)](https://huggingface.co/datasets/stanfordnlp/imdb) *(Subset used: 10,000 training samples, 2,000 testing samples)*

---

## Performance Evaluation Matrix

| Model Variant | Primary Metric (Precision / Recall / F1) | Generative Quality Metric (BLEU / ROUGE / Perplexity) | Training Time per Epoch | Key Observations / Constraints |
| :--- | :--- | :--- | :--- | :--- |
| **BERT Variant** | Accuracy: 0.881 <br> F1: ~0.880 | *N/A (Classification Task)* | ~4.5 mins | Excels at understanding bidirectional context. The `[CLS]` token efficiently summarizes document sentiment. |
| **GPT Variant** | *N/A (Generative Task)* | BLEU: 0.0 <br> ROUGE-1: ~0.187 <br> Perplexity: 43.86 | ~5.0 mins | Produces highly coherent, contextually accurate sequences. BLEU score of 0.0 is expected given the lack of exact 4-gram overlaps with the small reference set. |
| **Text-GAN Variant** | Discriminator Accuracy: ~0.89 | BLEU: 0.0 <br> ROUGE-L: ~0.0 | < 1.0 min | Generator struggles to form coherent syntax compared to autoregressive models. Highly susceptible to mode collapse. |

---

## Analytical Discussion

### Tokenization Differences
The three models utilize fundamentally different tokenization strategies. **BERT** relies on WordPiece subword tokenization, which allows it to handle out-of-vocabulary words effectively by breaking them into recognized prefixes and roots. It requires specific boundary tokens (`[CLS]` and `[SEP]`) to process text. **GPT** utilizes Byte-Level BPE (Byte Pair Encoding), granting it flexibility across raw text without a massive vocabulary limit, but requires a causal mask to prevent bidirectional lookahead during generation. The **Text-GAN**, utilizing a standard word-level or character-level embedding layer, lacks the pre-trained semantic clustering of the Transformer tokenizers, forcing it to learn syntax entirely from scratch.

### Metric Tradeoffs & Discrete Sequence Challenges
The generative metrics (BLEU/ROUGE) highlight a massive disparity between GPT and the Text-GAN. Generating text with a GAN introduces significant structural bottlenecks. In autoregressive models like GPT, the loss is directly computed using categorical cross-entropy. However, GANs use a Discriminator network to supply the loss gradient. Because text sequences are composed of discrete tokens, selecting a word via an `argmax` operation blocks backpropagation—the gradient becomes zero. 

While techniques like **Gumbel-Softmax** relaxation allow gradients to flow by creating a continuous probability distribution, the generator is essentially outputting "soft" probabilities rather than absolute text. This muddies the signal and causes the GAN to struggle with long-term temporal dependencies and discrete grammatical boundaries, leading to significantly lower ROUGE scores and frequent mode collapse compared to GPT.
