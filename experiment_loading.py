from transformers import AutoProcessor, Blip2ForConditionalGeneration

processor = AutoProcessor("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(
    "Salesforce/blip2-opt-2.7b", torch_dtype=torch.float16
)

print(model)
