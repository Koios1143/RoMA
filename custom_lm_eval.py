import logging
from lm_eval.api.registry import register_model
from lm_eval.models.huggingface import HFLM
from transformers.quantizers.auto import AutoQuantizationConfig
from transformers import OlmoeForCausalLM

import sys
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import torch
import transformers

eval_logger = logging.getLogger(__name__)
from lm_eval.models.utils import get_dtype

@register_model("roma")
class RoMAHFLM(HFLM):
    """
    Modified from HFLM. The only modification was made in the `_create_model` function.
    """

    def __init__(
            self,
            checkpoint_path: Optional[str] = None,
            **kwargs
    ):
        # Load config
        assert checkpoint_path is not None, "config_path is required"
        self.checkpoint_path = checkpoint_path
        super().__init__(**kwargs)

    def _create_model(
        self,
        pretrained: str,
        revision: str | None = "main",
        dtype: str | torch.dtype | None = "auto",
        trust_remote_code: bool | None = False,
        # arguments used for splitting a model across GPUs naively.
        # only used if `parallelize=True`.
        # (accelerate naive PP (device_map) options)
        parallelize: bool | None = False,
        gpus: int | None = None,
        max_memory_per_gpu: int | str | None = None,
        max_cpu_memory: int | str | None = None,
        offload_folder: str | None = "./offload",
        # PEFT, delta weights and quantization options
        peft: str | None = None,
        delta: str | None = None,
        autogptq: bool | str | None = False,
        gptqmodel: bool | None = False,
        gguf_file: str | None = None,
        quantization_config: AutoQuantizationConfig | None = None,
        subfolder: str = "",
        **kwargs,
    ) -> None:
        """
        Initializes an HF or HF-compatible PreTrainedModel from scratch
        inside HFLM, using the kwargs passed into self.__init__().

        Also handles functionality such as AutoGPTQ usage and PEFT wrapping.

        For future similar extensions to AutoGPTQ that are not core to HF's ecosystem,
        (such as PyTorch models that are nearly, but not quite, fully mirroring
        HF's public interface relied on in this HFLM class)
        please consider subclassing HFLM and overriding this and other methods as needed.
        """

        model_kwargs = kwargs if kwargs else {}

        model_kwargs.update(
            self._get_accelerate_args(
                parallelize=parallelize,
                device_map=kwargs.get("device_map", None),
                max_memory_per_gpu=max_memory_per_gpu,
                max_cpu_memory=max_cpu_memory,
                offload_folder=offload_folder,
                gpus=gpus,
            )
        )

        if model_kwargs.get("load_in_4bit", None):
            assert transformers.__version__ >= "4.30.0", (
                "load_in_4bit requires transformers >= 4.30.0"
            )
        if transformers.__version__ >= "4.30.0":
            if model_kwargs.get("load_in_4bit", None):
                if model_kwargs.get("bnb_4bit_compute_dtype", None):
                    model_kwargs["bnb_4bit_compute_dtype"] = get_dtype(
                        model_kwargs["bnb_4bit_compute_dtype"]
                    )
        
        # Create language model
        self._model = OlmoeForCausalLM.from_pretrained(pretrained, torch_dtype=torch.float16)

        # load gates
        payload = torch.load(self.checkpoint_path, map_location='cpu')
        target_layers = payload['target_layers']
        state = payload['state_dict']
        for li in target_layers:
            gate = self._model.model.layers[li].mlp.gate
            w_key = f'layers.{li}.mlp.gate.weight'
            b_key = f'layers.{li}.mlp.gate.bias'
            gate.weight.data.copy_(state[w_key])
            if gate.bias is not None and b_key in state:
                gate.bias.data.copy_(state[b_key])

        return None

    
import re
from lm_eval.__main__ import cli_evaluate

if __name__ == '__main__':
    """
    This script is revised based on one comment here:
    https://github.com/EleutherAI/lm-evaluation-harness/issues/1621#issuecomment-2400916502
    """
    # Get the arguments of the script
    sys.argv[0] = re.sub(r'(-script.pyw|.exe)?$', '', sys.argv[0])
    # Enter lm_harness_evaluation
    sys.exit(cli_evaluate())