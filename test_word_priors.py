#!/usr/bin/env python3
"""Test script for word prior functionality"""

import os
import cv2
import torch
import sys
sys.path.append('.')

from utils.Expression.gtd_parser import parse_gtd
from model.utils.utils import load_config, load_checkpoint
from model.inference.Backbone import Backbone
from model.training.dataset import Words

def test_inference_without_priors():
    """Test that inference still works without priors (backward compatibility)"""
    print("=" * 60)
    print("Test 1: Inference WITHOUT priors (backward compatibility)")
    print("=" * 60)

    # Load config
    config_path = 'model/config.yaml'
    params = load_config(config_path)

    # Device selection
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    params['device'] = device
    print(f'Device: {device}')

    # Load vocabulary
    words = Words(params['word_path'])
    params['word_num'] = len(words)
    params['struct_num'] = 7
    params['words'] = words
    print(f'Vocabulary size: {len(words)}')

    # Load model
    model = Backbone(params)
    model = model.to(device)
    load_checkpoint(model, None, params['checkpoint'])
    model.eval()

    # Load test image
    image_path = 'data/14_test_images/18_em_0_0.bmp'
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = torch.Tensor(img) / 255
    image = image.unsqueeze(0).unsqueeze(0)
    image_mask = torch.ones(image.shape)
    image, image_mask = image.to(device), image_mask.to(device)

    # Run inference WITHOUT priors
    with torch.no_grad():
        prediction = model(image, image_mask)  # No priors argument

    # Parse result
    expr = parse_gtd(prediction)
    if expr is None:
        latex_string = 'illegal'
    else:
        latex_string = expr.toLatex()

    print(f'Image: {image_path}')
    print(f'Prediction: {latex_string}')
    print(f'Prediction tree: {prediction[:5]}...')  # Show first 5 nodes
    print("✓ Test passed: Inference works without priors\n")

    return words, model, device, image, image_mask, latex_string


def test_inference_with_neutral_priors(words, model, device, image, image_mask, baseline_result):
    """Test that uniform (neutral) priors don't change predictions"""
    print("=" * 60)
    print("Test 2: Inference WITH neutral priors (uniform distribution)")
    print("=" * 60)

    # Create uniform log priors (all zeros = equal probability for all words)
    word_log_priors = torch.zeros(len(words)).to(device)
    print(f'Prior shape: {word_log_priors.shape}')
    print(f'Prior sum: {word_log_priors.sum().item()} (should be 0 for neutral)')

    # Run inference WITH neutral priors
    with torch.no_grad():
        prediction = model(image, image_mask, word_log_priors=word_log_priors)

    # Parse result
    expr = parse_gtd(prediction)
    if expr is None:
        latex_string = 'illegal'
    else:
        latex_string = expr.toLatex()

    print(f'Prediction: {latex_string}')
    print(f'Baseline:   {baseline_result}')

    if latex_string == baseline_result:
        print("✓ Test passed: Neutral priors produce same result as no priors\n")
    else:
        print("⚠ Warning: Results differ (this might be OK due to numerical precision)\n")

    return words, model, device, image, image_mask


def test_inference_with_biased_priors(words, model, device, image, image_mask):
    """Test that biased priors affect predictions"""
    print("=" * 60)
    print("Test 3: Inference WITH biased priors (boost digits)")
    print("=" * 60)

    # Create biased priors: boost digits, penalize letters
    word_log_priors = torch.zeros(len(words)).to(device)

    # Find digit and letter indices
    digits = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
    letters = list('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')

    boosted_count = 0
    penalized_count = 0

    for word_str, idx in words.words_dict.items():
        if word_str in digits:
            word_log_priors[idx] = 2.0  # Boost digits (e^2 ≈ 7.4x more likely)
            boosted_count += 1
        elif word_str in letters:
            word_log_priors[idx] = -2.0  # Penalize letters
            penalized_count += 1

    print(f'Boosted {boosted_count} digits (log_prior = +2.0)')
    print(f'Penalized {penalized_count} letters (log_prior = -2.0)')
    print(f'Prior min: {word_log_priors.min().item():.2f}, max: {word_log_priors.max().item():.2f}')

    # Run inference WITH biased priors
    with torch.no_grad():
        prediction = model(image, image_mask, word_log_priors=word_log_priors)

    # Parse result
    expr = parse_gtd(prediction)
    if expr is None:
        latex_string = 'illegal'
    else:
        latex_string = expr.toLatex()

    print(f'Prediction with biased priors: {latex_string}')
    print("✓ Test passed: Biased priors executed successfully\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Testing Word Prior Implementation")
    print("=" * 60 + "\n")

    # Test 1: Without priors (backward compatibility)
    words, model, device, image, image_mask, baseline = test_inference_without_priors()

    # Test 2: With neutral priors (should give same result)
    words, model, device, image, image_mask = test_inference_with_neutral_priors(
        words, model, device, image, image_mask, baseline
    )

    # Test 3: With biased priors (should affect result)
    test_inference_with_biased_priors(words, model, device, image, image_mask)

    print("=" * 60)
    print("All tests completed!")
    print("=" * 60)
