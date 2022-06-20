"""
unit tests
"""

import unittest

import torch
from models import CausalConv, Block, ConvDecoder, LanguageModel


class TestCausalConv(unittest.TestCase):
    def setUp(self):
        batch_size = 7
        dim = 8
        seq_len = 32
        kernel_size = 5
        self.conv = CausalConv(kernel_size, dim).eval()
        self.x = torch.randn(batch_size, dim, seq_len)

    def test_causality(self):
        y1, _ = self.conv(self.x)
        for k in [1, 3, 5, 7, 8]:
            y2, _ = self.conv(self.x[..., :-k])
            self.assertTrue((y1[..., :-k] == y2).all())

    def test_cache(self):
        y1, _ = self.conv.forward(self.x)
        for k in [0, 1, 5]:
            y2, _ = self.conv.forward(self.x[..., -k:], cache=self.x[..., :-k], return_cache=False)
            self.assertTrue((y1[..., -k:] == y2).all())

        y1, _ = self.conv.forward(self.x)
        for sz1, sz2 in [(20, 10), (18, 11), (8, 13)]:
            _, cache = self.conv.forward(self.x[..., :sz1], return_cache=True)
            y2, cache = self.conv.forward(self.x[..., sz1:(sz1 + sz2)], cache, return_cache=True)
            y3, _ = self.conv.forward(self.x[..., (sz1 + sz2):], cache, return_cache=False)
            self.assertTrue((y1[..., sz1:] == torch.cat((y2, y3), -1)).all())


class TestBlock(unittest.TestCase):
    def setUp(self):
        batch_size = 7
        dim = 8
        seq_len = 32
        kernel_size = 5
        self.block = Block(kernel_size, dim).eval()
        self.x = torch.randn(batch_size, dim, seq_len)

    def test_cache(self):
        y1, _ = self.block.forward(self.x)
        for sz1, sz2 in [(20, 10), (18, 11), (8, 13)]:
            _, cache = self.block.forward(self.x[..., :sz1], return_cache=True)
            y2, cache = self.block.forward(self.x[..., sz1:(sz1 + sz2)], cache, return_cache=True)
            y3, _ = self.block.forward(self.x[..., (sz1 + sz2):], cache, return_cache=False)
            self.assertTrue((y1[..., sz1:] == torch.cat((y2, y3), -1)).all())


class TestConvDecoder(unittest.TestCase):
    def setUp(self):
        batch_size = 7
        dim = 8
        seq_len = 32
        kernel_size = 5
        self.decoder = ConvDecoder(dim, kernel_size=kernel_size, depth_factor=None).eval()
        self.x = torch.randn(batch_size, dim, seq_len)

    def test_causality(self):
        y1, _ = self.decoder(self.x)
        for k in [1, 3, 5, 7, 8]:
            y2, _ = self.decoder(self.x[..., :-k])
            self.assertTrue((y1[..., :-k] == y2).all())

        y, _ = self.decoder(self.x)
        xx = self.x.clone()
        k = 5
        xx[..., 0, k] = 100000
        yy, _ = self.decoder(xx)
        self.assertTrue((y[..., :k] == yy[..., :k]).all())

    def test_long_range(self):
        y, _ = self.decoder(self.x)
        xx = self.x.clone()
        xx[..., 0, 0] = 100000
        yy, _ = self.decoder(xx)
        self.assertTrue((~torch.isclose(y[..., -1], yy[..., -1])).any())

    def test_shape(self):
        y, _ = self.decoder(self.x)
        self.assertTrue(y.shape == self.x.shape)

    def test_cache(self):
        for sz1, sz2 in [(20, 10), (18, 11), (8, 13)]:
            y1, _ = self.decoder.forward(self.x)
            _, cache = self.decoder.forward(self.x[..., :sz1], return_cache=True)
            y2, cache = self.decoder.forward(self.x[..., sz1:(sz1 + sz2)], cache, return_cache=True)
            y3, _ = self.decoder.forward(self.x[..., (sz1 + sz2):], cache, return_cache=False)
            assert (y1[..., sz1:] == torch.cat((y2, y3), -1)).all()


class TestLanguageModel(unittest.TestCase):
    def setUp(self):
        self.batch_size = 7
        dim = 8
        self.seq_len = 32
        kernel_size = 5
        self.num_tokens = 1000
        self.lm = LanguageModel(self.num_tokens, dim, kernel_size=kernel_size).eval()
        self.x = torch.randint(0, self.num_tokens, (self.batch_size, self.seq_len))

    def test_shape(self):
        logits, _, _ = self.lm(self.x)
        assert logits.shape == (self.batch_size, self.seq_len, self.num_tokens)

    def test_cache(self):
        for sz1, sz2 in [(20, 10), (18, 11), (8, 13)]:
            logits1, _, _ = self.lm(self.x)
            _, cache, _ = self.lm(self.x[..., :sz1], return_cache=True)
            logits2, cache, _ = self.lm(self.x[..., sz1:(sz1 + sz2)], cache=cache, return_cache=True)
            logits3, _, _ = self.lm(self.x[..., (sz1 + sz2):], cache=cache, return_cache=False)
            self.assertTrue(torch.isclose(logits1[..., sz1:, :], torch.cat((logits2, logits3), -2), rtol=1e-4).all())
