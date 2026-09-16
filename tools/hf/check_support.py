from vllm.model_executor.models.registry import ModelRegistry
from vllm.model_executor.layers.quantization import QUANTIZATION_METHODS

arches = set(ModelRegistry.get_supported_archs())
print("DeepseekV4ForCausalLM supported:", "DeepseekV4ForCausalLM" in arches)
print("deepseek arches present:", sorted(a for a in arches if "eepseek" in a))
print()
q = sorted(QUANTIZATION_METHODS)
print("quant methods (%d):" % len(q))
print("  ", ", ".join(q))
print()
print("auto-round present:", any("round" in x.lower() for x in q))
