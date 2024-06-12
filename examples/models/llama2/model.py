# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
from pathlib import Path
from collections import OrderedDict

import torch

from executorch.examples.models.llama2.llama_transformer import ModelArgs, Transformer

try:
    from .fairseq2 import convert_to_llama_checkpoint

except ImportError:

    def convert_to_llama_checkpoint(**kwargs):
        raise NotImplementedError(
            "Please install fairseq2 with `pip install fairseq2`."
        )


from ..model_base import EagerModelBase


BF16_LIST = ["odllm_0b5", "odllm_1b4", "llama_7b"]

# MODEL_NAME = "dummy_400k"
MODEL_NAME = "odllm_0b5"
# MODEL_NAME = "odllm_1b4"
# MODEL_NAME = "llama_7b"

# INPUTS = [1]
# INPUTS = [1,2,3]
MAX_INPUT_ID = 62
INPUTS = [i for i in range(1, MAX_INPUT_ID+1)]
# INPUTS[0] = INPUTS[len(INPUTS)-1]


class Llama2Model(EagerModelBase):
    def __init__(self, **kwargs):
        import pkg_resources

        # default path to the resource file
        # It currently supports 3 ways of specifying the checkpoint location:
        # 1. Using default path locates in examples/models/llama2/params
        # 2. Passing in the checkpoint path and params via kwargs
        # 3. Using the path from pkg_resources, only works with buck2
        try:
            # The 3rd way, if we can import this path, we are running with buck2, all resources can be accessed with pkg_resources.resource_filename
            # pyre-ignore
            from executorch.examples.models.llama2 import params

            ckpt_dir = Path(
                pkg_resources.resource_filename(
                    "executorch.examples.models.llama2", "params"
                )
            )
        except:
            # The 1st way
            ckpt_dir = Path(__file__).absolute().parent / "params"

        checkpoint_path = (
            kwargs["checkpoint"]
            if "checkpoint" in kwargs
            else ckpt_dir / f"{MODEL_NAME}.pth"
        )

        params_path = (
            kwargs["params"] if "params" in kwargs else ckpt_dir / f"{MODEL_NAME}.json"
        )

        self.use_kv_cache = (
            kwargs["use_kv_cache"] if "use_kv_cache" in kwargs else False
        )
        self.use_sdpa_with_kv_cache_op = (
            kwargs["use_sdpa_with_kv_cache"]
            if "use_sdpa_with_kv_cache" in kwargs
            else False
        )
        # The example is using a dummy small model with random weights for demo purpose only.
        # Follow the instruction in https://github.com/facebookresearch/llama to download the model
        device = "cpu"
        # if MODEL_NAME == "llama_7b":
        #   device = "meta"
        # flake8: noqa: TOR102
        checkpoint = torch.load(checkpoint_path, map_location=device, mmap=True)
        fairseq2_checkpoint = kwargs.get("fairseq2", False)
        if fairseq2_checkpoint:
            print("Using fairseq2 checkpoint")
            checkpoint = convert_to_llama_checkpoint(checkpoint=checkpoint)
        if "model" in checkpoint:
            # NB: some checkpoint contains a "model" field, which is the actual weights dict
            checkpoint = checkpoint["model"]

        if (not fairseq2_checkpoint) and checkpoint.get(
            "final_proj.weight", None
        ) is not None:
            print(
                """

************************************************************
This looks like a Fairseq2 checkpoint (based on the presence
of `final_proj.weight`.

You can import Fairseq2 checkpoints using the --fairseq2
option, but --fairseq2 was not specified.  Please verify
the checkpoint format to avoid generating faulty models.
************************************************************
"""
            )

        # # import pdb; pdb.set_trace()
        # # checkpoint = self.convert_bfloat16_to_float(checkpoint, MODEL_NAME)
        # checkpoint["tok_embeddings.weight"] = checkpoint["tok_embeddings.weight"].float()
        # # layers
        # # 0.5B -> 8 layers
        # # 1.4B -> 24 layers
        # num_layers = 8
        # if MODEL_NAME == "odllm_1b4":
        #     num_layers = 24

        # for i in range(num_layers):
        #     checkpoint[f"layers.{i}.attention.wq.weight"] = checkpoint[
        #         f"layers.{i}.attention.wq.weight"
        #     ].float()
        #     checkpoint[f"layers.{i}.attention.wk.weight"] = checkpoint[
        #         f"layers.{i}.attention.wk.weight"
        #     ].float()
        #     checkpoint[f"layers.{i}.attention.wv.weight"] = checkpoint[
        #         f"layers.{i}.attention.wv.weight"
        #     ].float()
        #     checkpoint[f"layers.{i}.attention.wo.weight"] = checkpoint[
        #         f"layers.{i}.attention.wo.weight"
        #     ].float()
        #     checkpoint[f"layers.{i}.feed_forward.w1.weight"] = checkpoint[
        #         f"layers.{i}.feed_forward.w1.weight"
        #     ].float()
        #     checkpoint[f"layers.{i}.feed_forward.w2.weight"] = checkpoint[
        #         f"layers.{i}.feed_forward.w2.weight"
        #     ].float()
        #     checkpoint[f"layers.{i}.feed_forward.w3.weight"] = checkpoint[
        #         f"layers.{i}.feed_forward.w3.weight"
        #     ].float()
        #     checkpoint[f"layers.{i}.attention_norm.weight"] = checkpoint[
        #         f"layers.{i}.attention_norm.weight"
        #     ].float()
        #     checkpoint[f"layers.{i}.ffn_norm.weight"] = checkpoint[
        #         f"layers.{i}.ffn_norm.weight"
        #     ].float()

        # checkpoint["norm.weight"] = checkpoint["norm.weight"].float()
        # checkpoint["output.weight"] = checkpoint["output.weight"].float()

        checkpoint = self.convert_bfloat16_to_float(checkpoint, MODEL_NAME)

        for idx in iter(checkpoint):
            cur = checkpoint[idx]
            if cur.dtype == torch.float16:
                checkpoint[idx] = checkpoint[idx].to(torch.float32)

        # get checkpoint dtype
        self.dtype = None
        if len(checkpoint) > 0:
            first = checkpoint[next(iter(checkpoint))]
            self.dtype = first.dtype
            mismatched_dtypes = [
                (key, value.dtype)
                for key, value in checkpoint.items()
                if value.dtype != self.dtype
            ]
            if len(mismatched_dtypes) > 0:
                print(
                    f"Mixed dtype model. Dtype of {first.key}: {first.dtype}. Mismatches in the checkpoint: {mismatched_dtypes}"
                )
        with open(params_path, "r") as f:
            params = json.loads(f.read())
        max_seq_len = 1024
        max_batch_size = 1
        model_args: ModelArgs = ModelArgs(
            max_seq_len=max_seq_len,
            max_batch_size=max_batch_size,
            use_kv_cache=self.use_kv_cache,
            use_sdpa_with_kv_cache_op=self.use_sdpa_with_kv_cache_op,
            eos_idx = len(INPUTS),
            **params,
        )
        model_parallel_size = 1
        n_kv_heads = model_args.n_heads if model_args.n_kv_heads is None else model_args.n_kv_heads
        n_local_heads = model_args.n_heads // model_parallel_size
        n_local_kv_heads = n_kv_heads // model_parallel_size
        n_rep = n_local_heads // n_local_kv_heads

        print(f"DX bos idx: {model_args.bos_idx}, eos idx: {model_args.eos_idx}; n_rep = {n_rep}")
        print(f"DX n_heads: {model_args.n_heads}, n_kv_heads: {model_args.n_kv_heads}")
        if kwargs.get("fairseq2", False):
            print("Using fairseq2 checkpoint")
            checkpoint = convert_to_llama_checkpoint(checkpoint=checkpoint)
        if kwargs.get("verbose", False):
            print("============= weights ================")
            print("{key} : {weights.numel()} : {weights.size()}")
            for key, weights in checkpoint.items():
                print(f"{key} : {weights.numel()} : {weights.size()}")
            print("============= /weights ================")

        # Within the device="meta" context, tensors that are created do not carry data.
        # They possess all other metadata a tensor carries such as size, stride, requires_grad.
        with torch.device("meta"):
            self.model_ = Transformer(model_args)

        if "int8" in str(checkpoint_path):
            print("Using int8 weight-only quantization!")
            from .quantize import WeightOnlyInt8QuantHandler

            simple_quantizer = WeightOnlyInt8QuantHandler(self.model_)
            self.model_ = simple_quantizer.convert_for_runtime()
        elif "int4" in str(checkpoint_path):
            print("Using int4 weight-only quantization!")
            from .quantize import Int8DynActInt4WeightQuantHandler

            simple_quantizer = Int8DynActInt4WeightQuantHandler(self.model_)
            self.model_ = simple_quantizer.convert_for_runtime()

        # assign=True: load params/buffers by assignment instead of performing an in-place copy.
        # Because we are using device="meta", tensors do not have memory associated with them
        # and an in-place copy is a no-op. Use assign=True in load_state_dict for this scenario.
        self.model_.load_state_dict(
            checkpoint,
            strict=False,
            assign=True,
        )  # self.model_ = Transformer(gptconf)

    def convert_bfloat16_to_float(self, checkpoint, model_name):
        output_ckpt = OrderedDict()
        output_ckpt["tok_embeddings.weight"] = checkpoint["tok_embeddings.weight"].to(torch.float32)
        # layers
        # 0.5B -> 8 layers
        # 1.4B -> 24 layers
        num_layers = 8
        if model_name == "odllm_1b4":
            num_layers = 24
        elif model_name == "llama_7b":
            num_layers = 32

        for i in range(num_layers):
            output_ckpt[f"layers.{i}.attention.wq.weight"] = checkpoint[
                f"layers.{i}.attention.wq.weight"
            ].to(torch.float32)
            output_ckpt[f"layers.{i}.attention.wk.weight"] = checkpoint[
                f"layers.{i}.attention.wk.weight"
            ].to(torch.float32)
            output_ckpt[f"layers.{i}.attention.wv.weight"] = checkpoint[
                f"layers.{i}.attention.wv.weight"
            ].to(torch.float32)
            output_ckpt[f"layers.{i}.attention.wo.weight"] = checkpoint[
                f"layers.{i}.attention.wo.weight"
            ].to(torch.float32)
            output_ckpt[f"layers.{i}.feed_forward.w1.weight"] = checkpoint[
                f"layers.{i}.feed_forward.w1.weight"
            ].to(torch.float32)
            output_ckpt[f"layers.{i}.feed_forward.w2.weight"] = checkpoint[
                f"layers.{i}.feed_forward.w2.weight"
            ].to(torch.float32)
            output_ckpt[f"layers.{i}.feed_forward.w3.weight"] = checkpoint[
                f"layers.{i}.feed_forward.w3.weight"
            ].to(torch.float32)
            output_ckpt[f"layers.{i}.attention_norm.weight"] = checkpoint[
                f"layers.{i}.attention_norm.weight"
            ].to(torch.float32)
            output_ckpt[f"layers.{i}.ffn_norm.weight"] = checkpoint[
                f"layers.{i}.ffn_norm.weight"
            ].to(torch.float32)

        output_ckpt["norm.weight"] = checkpoint["norm.weight"].to(torch.float32)
        output_ckpt["output.weight"] = checkpoint["output.weight"].to(torch.float32)
        return output_ckpt


    def get_eager_model(self):
        if self.dtype:
            # convert to the type of the provided checkpoint
            # input and output are torch.long, so signature unchanged
            return self.model_.to(self.dtype)
        else:
            # int8 quantization code has some bf16,
            # switch all to FP32
            return self.model_.to(torch.float32)

    def get_example_inputs(self):
        if self.use_kv_cache:
            return self.get_example_inputs_kvcache()
        else:
            return (
                torch.tensor(
                    [INPUTS], dtype=torch.long
                ),  # tokens, with kv cache our input token length is always just 1 token.
            )

    def get_example_inputs_kvcache(self):
        cache_sizes = self.model_.get_cache_sizes()
        cache_k = torch.zeros(cache_sizes, dtype=self.dtype)
        cache_v = torch.zeros(cache_sizes, dtype=self.dtype)
        return (
            torch.tensor(
                [[1]], dtype=torch.long
            ),  # tokens, with kv cache our input token length is always just 1 token.
            torch.tensor(
                0, dtype=torch.long
            ),  # start_pos, what token of output are we on.
            cache_k,  # key caches
            cache_v,  # value caches
        )
