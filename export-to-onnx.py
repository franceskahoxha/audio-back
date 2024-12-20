import torch
from encodec import EncodecModel

# Initialize the EnCodec model
model = EncodecModel.encodec_model_24khz()
model.eval()

# Dummy input for ONNX export
batch_size = 1
channels = 1
samples = 48000  # 1 second of 24kHz audio
dummy_input = torch.randn(batch_size, channels, samples)

# Define the wrapper
class EncodecModelWrapper(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        # Extract the encoded tensor
        encoded = self.model.encode(x)[0]  # Access the tensor from the tuple
        
        # Create a placeholder for scales with the same shape as the encoded tensor
        scales = torch.ones_like(encoded[0])  # `encoded[0]` is the actual tensor

        return encoded, scales

# Wrap the model
wrapped_model = EncodecModelWrapper(model)

# Export the ONNX model
torch.onnx.export(
    wrapped_model,
    dummy_input,
    "encodec_model.onnx",
    export_params=True,
    opset_version=13,
    input_names=["input"],
    output_names=["encoded_audio", "audio_scales"],
    dynamic_axes={"input": {0: "batch_size", 2: "samples"}}
)

print("Model exported to encodec_model.onnx")
