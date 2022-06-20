import torch
import torch.nn as nn

get_or_none = lambda d, key: d.get(key, None)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8, axis=-1):
        nn.Module.__init__(self)

        self.eps = eps
        self.axis = axis
        self.dim = dim

        self.dim_norm = dim ** (-1. / 2)

        shape = axis * (1,) + (dim,) + (-axis) * (1,)
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        shape = [1, ] * x.ndim
        shape[self.axis] = self.dim
        scale = self.scale.view(shape)

        rms = self.dim_norm * x.norm(dim=self.axis, keepdim=True)
        x_normed = x / (rms + self.eps)
        return scale * x_normed


class CausalConv(nn.Module):
    def __init__(self, kernel_size, dim, groups=1):
        nn.Module.__init__(self)

        assert kernel_size % 2 == 1
        assert dim % groups == 0

        self.kernel_size = kernel_size
        self.dim = dim

        self.context_size = kernel_size - 1

        self.pad_param = nn.Parameter(torch.randn((dim, self.context_size)))

        self.conv = nn.Conv1d(in_channels=dim, out_channels=dim, kernel_size=kernel_size, groups=groups)

    def pad(self, x, sz=None):
        if sz is None:
            sz = self.context_size
        assert sz <= self.context_size
        # return nn.functional.pad(x, [sz, 0])
        pad_param = self.pad_param[:, :sz].view(*(x.ndim - 2) * (1,), self.dim, -1).expand(*x.shape[:-2], -1, -1)
        return torch.cat((pad_param, x), -1)

    def forward_no_cache(self, x, return_cache=False):
        x = self.pad(x)
        y = self.conv(x)
        if return_cache:
            for_cache = x[..., -self.context_size:].clone()
        else:
            for_cache = None
        return y, for_cache

    def forward_cache(self, x, cache, return_cache=True):
        assert x.size(-1) > 0
        if cache is None:
            cache = torch.empty(x.shape[:-1] + (0,), device=x.device)
        if cache.size(-1) < self.context_size:
            cache = self.pad(cache, self.context_size - cache.size(-1))
        else:
            cache = cache[..., -self.context_size:]

        x = torch.cat((cache, x), -1)
        y = self.conv(x)
        if return_cache:
            cache = x[..., -self.context_size:].clone()
        else:
            cache = None
        return y, cache

    def forward(self, x, cache=None, return_cache=False):
        if cache is not None:
            return self.forward_cache(x, cache, return_cache)
        else:
            return self.forward_no_cache(x, return_cache)


class Block(nn.Module):
    def __init__(self, kernel_size, dim, groups=1, dropout=0.0):
        nn.Module.__init__(self)

        self.kernel_size = kernel_size
        self.dropout = dropout

        conv = lambda: CausalConv(kernel_size, dim, groups)
        norm = lambda: RMSNorm(dim, axis=-2)

        self.conv1 = conv()
        self.norm1 = norm()
        self.conv2 = conv()
        self.norm2 = norm()

        self.context_size = self.conv1.context_size + self.conv2.context_size

    def forward(self, x, cache=None, return_cache=False):
        if cache is not None:
            cache = cache.copy()  # shallow copy!
        else:
            cache = dict()

        gelu = nn.functional.gelu

        x0 = x

        x1, cache['pre_conv1'] = self.conv1.forward(x0, get_or_none(cache, 'pre_conv1'), return_cache)
        x2 = self.norm1(x1)
        x3 = gelu(x2)
        if self.training and (self.dropout > 0):
            nn.functional.dropout(x3, p=self.dropout, inplace=True)

        x4, cache['pre_conv2'] = self.conv2.forward(x3, get_or_none(cache, 'pre_conv2'), return_cache)
        x5 = self.norm2(x4)

        x6 = x5 + x0
        x7 = gelu(x6)
        if self.training and (self.dropout > 0):
            nn.functional.dropout(x7, p=self.dropout, inplace=True)

        if not return_cache:
            cache = None
        return x7, cache


class ConvDecoder(nn.Module):
    def __init__(self, dim=32, num_low_level_blocks=3, num_recurring_blocks=2, kernel_size=5,
                 depth_factor=1, dropout=0.1):
        nn.Module.__init__(self)

        conv_block = lambda: Block(kernel_size, dim, dropout=dropout)

        self.low_level_blocks = nn.ModuleList()
        for i in range(num_low_level_blocks):
            self.low_level_blocks.append(conv_block())

        self.recurring_blocks = nn.ModuleList()
        recurring_context_size = 0
        for i in range(num_recurring_blocks):
            block = conv_block()
            self.recurring_blocks.append(block)
            recurring_context_size += block.context_size

        if depth_factor is None:
            depth_factor = recurring_context_size // 2
        assert depth_factor <= recurring_context_size
        self.depth_factor = depth_factor

    def forward(self, x, cache=None, return_cache=False):
        if cache is not None:
            cache = cache.copy()  # shallow copy!
        else:
            cache = dict()

        for i, block in enumerate(self.low_level_blocks):
            name = f'low_level_{i:d}'
            x, cache[name] = block(x, get_or_none(cache, name), return_cache)

        cache['seq_len'] = cache.get('seq_len', 0)
        seq_len = x.size(-1) + cache['seq_len']
        for i in range(0, seq_len, self.depth_factor):
            for j, block in enumerate(self.recurring_blocks):
                name = f'recurring_level_{i:d}_{j:d}'
                x[..., -(seq_len - i):], cache[name] = block(x[..., -(seq_len - i):], get_or_none(cache, name),
                                                             return_cache)
        cache['seq_len'] = seq_len

        if not return_cache:
            cache = None
        return x, cache


class LanguageModel(nn.Module):
    def __init__(self, num_tokens, dim=32, num_low_level_blocks=3, num_recurring_blocks=2, kernel_size=5, depth_factor=None, dropout=0.1):
        nn.Module.__init__(self)

        self.embedding = nn.Embedding(num_tokens, dim)
        self.decoder = ConvDecoder(dim, num_low_level_blocks, num_recurring_blocks, kernel_size, depth_factor, dropout)
        self.norm = RMSNorm(dim, axis=-2)
        self.proj = nn.Linear(dim, num_tokens)  # consider weight tying with embedding...

        self.init_parameters()

    def init_parameters(self):
        with torch.no_grad():
            for name, p in self.named_parameters():
                if 'bias' in name:
                    p.data[:] = 0.0
                else:
                    p.normal_(0.0, 0.02)

    def forward(self, x, targets=None, cache=None, return_cache=False):
        x = self.embedding(x).transpose(-2, -1)
        x, cache = self.decoder(x, cache, return_cache)
        x = self.norm(x)
        x.transpose_(-2, -1)
        logits = self.proj(x)
        if not return_cache:
            cache = None
        if targets is not None:
            loss = nn.functional.cross_entropy(x.view(-1, x.size(-1)), targets.view(-1))
        else:
            loss = None
        return logits, cache, loss
