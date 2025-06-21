# Tokenise the following text:
from transformers import GPT2Tokenizer, BertTokenizer, T5Tokenizer
import sentencepiece as spm

# Tokenise the following sentence using BPE, WordPiece, and SentencePiece (Unigram)
text = "The cat sat on the mat because it was tired."

print(f"Original text: {text}")
print("=" * 80)

# 1. BPE (Byte Pair Encoding) using GPT-2 tokenizer
print("1. BPE (Byte Pair Encoding) - GPT-2 Tokenizer")
print("-" * 50)
bpe_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
bpe_tokens = bpe_tokenizer.tokenize(text)
bpe_token_ids = bpe_tokenizer.encode(text)

print(f"Tokens: {bpe_tokens}")
print(f"Token IDs: {bpe_token_ids}")
print(f"Total token count: {len(bpe_tokens)}")
print(f"Note: 'Ġ' represents spaces in GPT-2 tokenization")
print()

# 2. WordPiece using BERT tokenizer
print("2. WordPiece - BERT Tokenizer")
print("-" * 50)
wordpiece_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
wordpiece_tokens = wordpiece_tokenizer.tokenize(text)
wordpiece_token_ids = wordpiece_tokenizer.encode(text, add_special_tokens=False)

print(f"Tokens: {wordpiece_tokens}")
print(f"Token IDs: {wordpiece_token_ids}")
print(f"Total token count: {len(wordpiece_tokens)}")
print()

# 3. SentencePiece (Unigram) using T5 tokenizer
print("3. SentencePiece (Unigram) - T5 Tokenizer")
print("-" * 50)
sentencepiece_tokenizer = T5Tokenizer.from_pretrained('t5-small')
sentencepiece_tokens = sentencepiece_tokenizer.tokenize(text)
sentencepiece_token_ids = sentencepiece_tokenizer.encode(text, add_special_tokens=False)

print(f"Tokens: {sentencepiece_tokens}")
print(f"Token IDs: {sentencepiece_token_ids}")
print(f"Total token count: {len(sentencepiece_tokens)}")
print(f"Note: '▁' represents spaces in SentencePiece tokenization")
print()


# MASK & PREDICT SECTION
print("\n" + "=" * 80)
print("MASK & PREDICT with Fill-Mask Pipeline")
print("=" * 80)

from transformers import pipeline
import torch
import json

# Use the largest available fill-mask model
print("Loading large fill-mask model...")
model_name = "microsoft/deberta-v3-large"

# Container for storing predictions
predictions_dict = {
    "model_used": model_name,
    "approach_1_multiple_masks": [],
    "approach_2_single_masks": {
        "first_mask": [],
        "second_mask": []
    }
}

try:
    fill_mask = pipeline(
        "fill-mask",
        model=model_name,
        device=0 if torch.cuda.is_available() else -1
    )
    
    print("=" * 50)
    print("APPROACH 1: Multiple Masks Simultaneously")
    print("=" * 50)
    
    masked_text_multiple = "The cat sat on the [MASK] because it was [MASK]."
    print(f"Masked text: {masked_text_multiple}")
    
    predictions_multiple = fill_mask(masked_text_multiple, top_k=3)
    
    # Handle multiple masks - returns list of lists
    print(f"\nPredictions for multiple masks:")
    if isinstance(predictions_multiple[0], list):
        # Multiple masks return format: [[mask1_predictions], [mask2_predictions]]
        for mask_idx, mask_predictions in enumerate(predictions_multiple):
            print(f"\nMask {mask_idx + 1} predictions:")
            for i, pred in enumerate(mask_predictions, 1):
                entry = {
                    "mask_position": mask_idx + 1,
                    "rank": i,
                    "token": pred['token_str'].strip(),
                    "score": round(pred['score'], 4),
                    "sequence": pred['sequence']
                }
                predictions_dict["approach_1_multiple_masks"].append(entry)
                print(f"{i}. Token: '{entry['token']}' | Score: {entry['score']}")
                print(f"   Complete sentence: {entry['sequence']}")
    else:
        # Fallback for single mask format
        for i, pred in enumerate(predictions_multiple, 1):
            entry = {
                "rank": i,
                "token": pred['token_str'].strip(),
                "score": round(pred['score'], 4),
                "sequence": pred['sequence']
            }
            predictions_dict["approach_1_multiple_masks"].append(entry)
            print(f"{i}. Token: '{entry['token']}' | Score: {entry['score']}")
            print(f"   Complete sentence: {entry['sequence']}")
    
    print("=" * 50)
    print("APPROACH 2: Single Masks Separately")
    print("=" * 50)

    # First mask
    masked_text_1 = "The cat sat on the [MASK] because it was tired."
    print(f"\nProcessing first mask: {masked_text_1}")
    predictions_1 = fill_mask(masked_text_1, top_k=3)
    
    print(f"Top 3 predictions for first [MASK]:")
    for i, pred in enumerate(predictions_1, 1):
        entry = {
            "rank": i,
            "token": pred['token_str'].strip(),
            "score": round(pred['score'], 4),
            "sequence": pred['sequence']
        }
        predictions_dict["approach_2_single_masks"]["first_mask"].append(entry)
        print(f"{i}. Token: '{entry['token']}' | Score: {entry['score']}")
        print(f"   Complete sentence: {entry['sequence']}")

    # Second mask
    masked_text_2 = "The cat sat on the mat because it was [MASK]."
    print(f"\nProcessing second mask: {masked_text_2}")
    predictions_2 = fill_mask(masked_text_2, top_k=3)
    
    print(f"Top 3 predictions for second [MASK]:")
    for i, pred in enumerate(predictions_2, 1):
        entry = {
            "rank": i,
            "token": pred['token_str'].strip(),
            "score": round(pred['score'], 4),
            "sequence": pred['sequence']
        }
        predictions_dict["approach_2_single_masks"]["second_mask"].append(entry)
        print(f"{i}. Token: '{entry['token']}' | Score: {entry['score']}")
        print(f"   Complete sentence: {entry['sequence']}")
    
    print(f"\n📋 COMPARISON ANALYSIS:")
    print(f"Multiple Masks: The model predicts both positions simultaneously, but assumes")
    print(f"independence between them. This can lead to less contextually appropriate combinations.")
    print(f"\nSingle Masks: Each prediction uses the full context of the other words,")
    print(f"leading to more coherent and contextually appropriate predictions.")
    print(f"\nModel used: {model_name} (DeBERTa-v3-large with ~435M parameters)")

except Exception as e:
    print(f"Error with fill-mask pipeline: {e}")
    print("Falling back to smaller model...")
    
    try:
        model_name_fallback = "bert-large-uncased"
        predictions_dict["model_used"] = model_name_fallback
        print(f"Using fallback model: {model_name_fallback}")

        fill_mask = pipeline("fill-mask", model=model_name_fallback)

        # Multiple masks with fallback
        masked_text_multiple = "The cat sat on the [MASK] because it was [MASK]."
        predictions_multiple = fill_mask(masked_text_multiple, top_k=3)
        
        print(f"\nMultiple masks approach:")
        if isinstance(predictions_multiple[0], list):
            # Multiple masks format
            for mask_idx, mask_predictions in enumerate(predictions_multiple):
                print(f"Mask {mask_idx + 1}:")
                for i, pred in enumerate(mask_predictions, 1):
                    entry = {
                        "mask_position": mask_idx + 1,
                        "rank": i,
                        "token": pred['token_str'].strip(),
                        "score": round(pred['score'], 4),
                        "sequence": pred['sequence']
                    }
                    predictions_dict["approach_1_multiple_masks"].append(entry)
                    print(f"  {i}. {entry['token']} (score: {entry['score']})")
        else:
            # Single mask format fallback
            for i, pred in enumerate(predictions_multiple, 1):
                entry = {
                    "rank": i,
                    "token": pred['token_str'].strip(),
                    "score": round(pred['score'], 4),
                    "sequence": pred['sequence']
                }
                predictions_dict["approach_1_multiple_masks"].append(entry)
                print(f"{i}. {entry['token']} (score: {entry['score']})")

        # Single masks with fallback
        masked_text_1 = "The cat sat on the [MASK] because it was tired."
        masked_text_2 = "The cat sat on the mat because it was [MASK]."
        
        predictions_1 = fill_mask(masked_text_1, top_k=3)
        predictions_2 = fill_mask(masked_text_2, top_k=3)

        print(f"\nSingle mask approach - First [MASK]:")
        for i, pred in enumerate(predictions_1, 1):
            entry = {
                "rank": i,
                "token": pred['token_str'].strip(),
                "score": round(pred['score'], 4),
                "sequence": pred['sequence']
            }
            predictions_dict["approach_2_single_masks"]["first_mask"].append(entry)
            print(f"{i}. {entry['token']} (score: {entry['score']})")

        print(f"\nSingle mask approach - Second [MASK]:")
        for i, pred in enumerate(predictions_2, 1):
            entry = {
                "rank": i,
                "token": pred['token_str'].strip(),
                "score": round(pred['score'], 4),
                "sequence": pred['sequence']
            }
            predictions_dict["approach_2_single_masks"]["second_mask"].append(entry)
            print(f"{i}. {entry['token']} (score: {entry['score']})")

    except Exception as e2:
        print(f"Fallback also failed: {e2}")
        print("Please ensure you have sufficient memory and the required dependencies installed.")
        print("Try: pip install transformers torch")
        predictions_dict["error"] = str(e2)

# Write predictions to JSON file
output_file = "predictions.json"
with open(output_file, "w") as f:
    json.dump(predictions_dict, f, indent=4)

print(f"\n✅ Predictions saved to: {output_file}")
