# Tokenization and Mask Prediction Analysis

A comprehensive study of different tokenization algorithms and masked language model predictions.

## Overview

This project demonstrates and compares three different tokenization algorithms:
- **BPE (Byte Pair Encoding)** using GPT-2 tokenizer
- **WordPiece** using BERT tokenizer
- **SentencePiece (Unigram)** using T5 tokenizer

Additionally, it implements masked language modeling with fill-mask capabilities using large transformer models.

## Features

### 1. Tokenization Analysis
- Compares three major tokenization algorithms on the same input text
- Shows token IDs, token counts, and algorithm-specific characteristics
- Generates detailed comparison analysis in `compare.md`

### 2. Masked Language Modeling
- Two approaches to mask prediction:
  - **Multiple masks simultaneously**: Predicts both positions at once
  - **Single masks separately**: Predicts each position independently
- Uses large transformer models (DeBERTa-v3-large as primary, BERT-large as fallback)
- Saves structured predictions to `predictions.json`

## Requirements

### Dependencies
```bash
pip install transformers
pip install sentencepiece
pip install torch
```

### Hardware
- GPU support optional (CUDA) but recommended for faster inference
- CPU execution supported as fallback

## Usage

### Running the Script
```bash
cd q1
python tokenise.py
```

### Input Text
The script analyzes the sentence: *"The cat sat on the mat because it was tired."*

## Output Files

### 1. `compare.md`
Contains detailed comparison of tokenization algorithms including:
- How each algorithm handles whitespace
- Subword segmentation strategies
- Vocabulary construction methods
- Key differences and use cases

### 2. `predictions.json`
Structured JSON containing:
```json
{
    "model_used": "microsoft/deberta-v3-large",
    "approach_1_multiple_masks": [...],
    "approach_2_single_masks": {
        "first_mask": [...],
        "second_mask": [...]
    }
}
```

Each prediction includes:
- Token rank and confidence score
- Predicted token text
- Complete sentence with prediction

## Key Observations

### Tokenization Differences

| Algorithm | Space Representation | Approach | Token Count* |
|-----------|---------------------|----------|--------------|
| BPE (GPT-2) | `Ġ` prefix | Merge frequent pairs | 11 |
| WordPiece (BERT) | `##` continuation | Linguistically aware | 11 |
| SentencePiece (T5) | `▁` prefix | Probabilistic segmentation | 13 |

*Token count depends on vocabulary and training data

### Mask Prediction Analysis

#### Multiple Masks vs Single Masks
- **Multiple masks**: Processes both `[MASK]` tokens simultaneously but assumes independence
- **Single masks**: Uses full context for each prediction, generally more coherent

#### Model Performance
- Primary model: **microsoft/deberta-v3-large** (~435M parameters)
- Fallback model: **bert-large-uncased** (~340M parameters)
- Error handling includes automatic fallback to smaller models

## Technical Details

### Tokenization Process
1. Loads pre-trained tokenizers from Hugging Face
2. Tokenizes input text using each algorithm
3. Extracts tokens, token IDs, and counts
4. Analyzes and compares characteristics

### Mask Prediction Process
1. Creates masked versions of input text
2. Loads large transformer models via pipeline
3. Generates top-k predictions (k=3)
4. Handles both single and multiple mask scenarios
5. Structures output with confidence scores

### Error Handling
- Unicode character handling (smart quotes → ASCII)
- Model availability checking
- GPU/CPU device selection
- Graceful fallback to smaller models

## Expected Console Output

The script provides real-time feedback including:
- Tokenization results for each algorithm
- Model loading status
- Top predictions with confidence scores
- Comparative analysis summary

## Troubleshooting

### Common Issues
1. **Unicode errors**: Ensure text uses standard ASCII quotes
2. **Model download**: First run requires internet for model downloads
3. **Memory issues**: Use CPU if GPU memory insufficient
4. **Model availability**: Script automatically falls back to available models

### Performance Notes
- Initial run downloads models (~1-2GB total)
- GPU acceleration significantly faster for large models
- Predictions quality varies with model size and training data

## Model Architecture Notes

### DeBERTa-v3-Large
- **Parameters**: ~435M
- **Architecture**: Enhanced BERT with disentangled attention
- **Strengths**: Superior contextual understanding
- **Use case**: Primary model for high-quality predictions

### BERT-Large-Uncased
- **Parameters**: ~340M
- **Architecture**: Bidirectional transformer
- **Strengths**: Reliable, well-established
- **Use case**: Fallback model for compatibility
