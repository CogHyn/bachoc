from KG_builder.embedding.load.cost import GeminiEmbedModel
from KG_builder.embedding.load.free import QwenEmbedding

qwen: QwenEmbedding = QwenEmbedding(model_name="Qwen/Qwen2.5-0.5B-Instruct")