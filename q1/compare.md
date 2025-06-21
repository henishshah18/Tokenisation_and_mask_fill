**Comparison of Tokenizers**

BPE (Byte Pair Encoding): Uses 'Ġ' to represent spaces and learns subword units by 
iteratively merging frequent character pairs. It creates longer subwords and handles 
out-of-vocabulary words by breaking them into learnable pieces.

WordPiece: It produces linguistically meaningful subwords and handles prefixes/suffixes better, 
leading to different tokenization patterns than BPE.

SentencePiece (Unigram): Uses '▁' for spaces and employs a probabilistic approach 
to find optimal segmentation. It can split words more aggressively and doesn't rely 
on pre-tokenization, making it language-agnostic but often producing more tokens.

Key differences: Each algorithm handles whitespace differently (Ġ vs ## vs ▁), uses 
different merging strategies, and has varying vocabulary construction methods, leading 
to different granularities in text segmentation.