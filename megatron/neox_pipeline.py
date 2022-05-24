import copy
import torch

from deepspeed.runtime.pipe.engine import PipelineEngine

from megatron import mpu
from megatron.utils import print_rank_0, setup_for_inference_or_eval
from megatron.text_generation_utils import generate_samples_from_prompt


neox20_base_config = {
    "checkpoint_model_parallel_size": 2,
    "is_pipe_parallel": True,
    "pipe_parallel_size": 1,
    "train_micro_batch_size_per_gpu": 1,
    "disable_data_helper": True,
    "gradient_accumulation_steps": 1,
    "fp16": {"enabled": True},
    "num_layers": 44,
    "hidden_size": 6144,
    "num_attention_heads": 64,
    "seq_length": 2048,
    "max_position_embeddings": 2048,
    "pos_emb": "rotary",
    "no_weight_tying": True,
    "scaled_upper_triang_masked_softmax_fusion": True,
    "bias_gelu_fusion": True,
    "rotary_pct": 0.25,
    "init_method": "small_init",
    "output_layer_init_method": "wang_init",
    "gpt_j_residual": True,
    "output_layer_parallelism": "column",
    "tokenizer_type": "HFTokenizer",
    "split": "995,4,1",
    "attention_dropout": 0,
    "hidden_dropout": 0,
    "synchronize_each_layer": True,
    "text_gen_type": "unconditional"
}


class NeoXPipeline():
    def __init__(self, config):
        megatron_config = copy.deepcopy(neox20_base_config)
        megatron_config.update(config)

        model, neox_args = setup_for_inference_or_eval(use_cache=True, megatron_config=megatron_config)
        self.model = model
        self.neox_args = neox_args
        self.megatron_config = megatron_config
        self.latencies = []

    def _call(self, request, do_sample=True, min_length=50):
        if not isinstance(self.model, PipelineEngine):
            self.model = self.model.module
        generated_texts = generate_samples_from_prompt(
            neox_args=self.neox_args,
            model=self.model,
            text=request,
            maximum_tokens=min_length
        )
        if torch.distributed.get_rank() == 0:
            self.latencies.append(generated_texts[0]['duration_seconds'])
        return generated_texts
    
    __call__ = _call

