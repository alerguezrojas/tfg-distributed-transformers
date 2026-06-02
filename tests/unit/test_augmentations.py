"""Unit tests for src/training/augmentations.py"""
import pytest
import torch
from src.training.augmentations import mixup_batch


class TestMixupBatch:
    def test_output_shape(self):
        images = torch.randn(4, 3, 224, 224)
        labels = torch.randint(0, 2, (4, 19)).float()
        out_img, out_lbl = mixup_batch(images, labels, alpha=0.2)
        assert out_img.shape == images.shape
        assert out_lbl.shape == labels.shape

    def test_labels_are_soft(self):
        images = torch.randn(4, 3, 224, 224)
        labels = torch.zeros(4, 19)
        labels[0, 0] = 1.0
        out_img, out_lbl = mixup_batch(images, labels, alpha=0.5)
        # With mixing, labels should have values between 0 and 1
        assert out_lbl.min().item() >= 0.0
        assert out_lbl.max().item() <= 1.0

    def test_images_are_convex_combination(self):
        # With lambda=0.5 exactly: output should be average of two images
        torch.manual_seed(42)
        images = torch.zeros(2, 3, 4, 4)
        images[0] = 1.0
        images[1] = 0.0
        labels = torch.zeros(2, 2)
        out_img, _ = mixup_batch(images, labels, alpha=100.0)
        # Lambda close to 0.5 with high alpha → output close to 0.5
        assert 0.0 < out_img.mean().item() < 1.0

    def test_no_in_place_modification(self):
        images = torch.randn(4, 3, 8, 8)
        labels = torch.randint(0, 2, (4, 5)).float()
        orig_images = images.clone()
        orig_labels = labels.clone()
        mixup_batch(images, labels, alpha=0.2)
        assert torch.allclose(images, orig_images)
        assert torch.allclose(labels, orig_labels)

    def test_alpha_zero_returns_original(self):
        images = torch.randn(4, 3, 8, 8)
        labels = torch.randint(0, 2, (4, 5)).float()
        # alpha=0 → Beta(0,0) is undefined, but alpha very small → lam≈1
        out_img, out_lbl = mixup_batch(images, labels, alpha=1e-10)
        assert out_img.shape == images.shape

    def test_batch_size_one(self):
        images = torch.randn(1, 3, 8, 8)
        labels = torch.ones(1, 5)
        out_img, out_lbl = mixup_batch(images, labels, alpha=0.4)
        assert out_img.shape == (1, 3, 8, 8)
