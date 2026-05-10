from functools import wraps
from collections import namedtuple

import os
import re
import torch
from torch import nn, einsum
import torch.nn.functional as F


FlashAttentionConfig = namedtuple('FlashAttentionConfig', ['enable_flash', 'enable_math', 'enable_mem_efficient'])
_VERSION_RE = re.compile(r'\d+')


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def major_minor(version_string):
    numbers = [int(match.group(0)) if (match := _VERSION_RE.match(part)) else 0 for part in version_string.split('+', 1)[0].split('.')[:2]]
    return tuple(numbers + [0] * (2 - len(numbers)))


def once(fn):
    called = False

    @wraps(fn)
    def inner(x):
        nonlocal called
        if called:
            return
        called = True
        return fn(x)
    return inner

print_once = once(print)


class Attend(nn.Module):
    def __init__(
        self,
        dropout = 0.,
        flash = False,
        scale = None
    ):
        super().__init__()
        self.scale = scale
        self.dropout = dropout
        self.attn_dropout = nn.Dropout(dropout)

        self.flash = flash
        assert not (flash and major_minor(torch.__version__) < (2, 0)), 'in order to use flash attention, you must be using pytorch 2.0 or above'

        self.cpu_config = FlashAttentionConfig(True, True, True)
        self.cuda_config = None

        if not torch.cuda.is_available() or not flash:
            return

        device_properties = torch.cuda.get_device_properties(torch.device('cuda'))

        if (device_properties.major, device_properties.minor) >= (8, 0):
            if os.name == 'nt':
                print_once('Windows OS detected, using math or mem efficient attention if input tensor is on cuda')
                self.cuda_config = FlashAttentionConfig(False, True, True)
            else:
                print_once('GPU Compute Capability equal or above 8.0, using flash attention if input tensor is on cuda')
                self.cuda_config = FlashAttentionConfig(True, False, False)
        else:
            print_once('GPU Compute Capability below 8.0, using math or mem efficient attention if input tensor is on cuda')
            self.cuda_config = FlashAttentionConfig(False, True, True)

    def flash_attn(self, q, k, v):
        if exists(self.scale):
            q = q * (self.scale / (q.shape[-1] ** -0.5))

        with torch.backends.cuda.sdp_kernel(**(self.cuda_config if q.is_cuda else self.cpu_config)._asdict()):
            return F.scaled_dot_product_attention(q, k, v, dropout_p=self.dropout if self.training else 0.)

    def forward(self, q, k, v):
        scale = default(self.scale, q.shape[-1] ** -0.5)

        if self.flash:
            return self.flash_attn(q, k, v)

        sim = einsum("b h i d, b h j d -> b h i j", q, k) * scale
        return einsum("b h i j, b h j d -> b h i d", self.attn_dropout(sim.softmax(dim=-1)), v)
