# coding=utf-8
# Copyright 2022 The OFA-Sys Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Tokenization classes for OFA."""
from transformers.utils import logging
from transformers import BertTokenizerFast

from typing import Union
import os

logger = logging.get_logger(__name__)


class OFATokenizerFastForChinese(BertTokenizerFast):

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], *init_inputs, **kwargs):
        # 中文使用BertTokenizerFast
        bos_token = "<s>"
        eos_token = "</s>"
        sep_token = "</s>"
        cls_token = "<s>"
        unk_token = "<unk>"
        pad_token = "<pad>"
        mask_token = "<mask>"
        tokenizer = super().from_pretrained(
            pretrained_model_name_or_path, bos_token=bos_token, eos_token=eos_token, sep_token=sep_token,
            cls_token=cls_token, unk_token=unk_token, pad_token=pad_token, mask_token=mask_token, *init_inputs, **kwargs
        )
        tokenizer.add_tokens(["<code_{}>".format(i) for i in range(8192)])
        tokenizer.add_tokens(["<bin_{}>".format(i) for i in range(1000)])
        return tokenizer
