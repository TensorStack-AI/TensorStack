using System.Text.Json.Serialization;

namespace TensorStack.Python.Config
{
    public sealed record CheckpointConfig
    {
        [JsonPropertyName("model_checkpoint")]
        public string ModelCheckpoint { get; set; }

        [JsonPropertyName("vae_checkpoint")]
        public string VaeCheckpoint { get; set; }

        [JsonPropertyName("text_encoder_checkpoint")]
        public string TextEncoderCheckpoint { get; set; }
    }
}
