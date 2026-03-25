using System.Text.Json.Serialization;

namespace TensorStack.Python.Config
{
    public sealed record CheckpointConfig
    {

        [JsonPropertyName("single_file")]
        public string SingleFile { get; set; }

        [JsonPropertyName("text_encoder")]
        public string TextEncoder { get; set; }

        [JsonPropertyName("text_encoder_2")]
        public string TextEncoder2 { get; set; }

        [JsonPropertyName("text_encoder_3")]
        public string TextEncoder3 { get; set; }

        [JsonPropertyName("transformer")]
        public string Transformer { get; set; }

        [JsonPropertyName("transformer_2")]
        public string Transformer2 { get; set; }

        [JsonPropertyName("vae")]
        public string Vae { get; set; }

        [JsonPropertyName("audio_vae")]
        public string AudioVae { get; set; }

        [JsonPropertyName("vocoder")]
        public string Vocoder { get; set; }

        [JsonPropertyName("connectors")]
        public string Connectors { get; set; }
    }
}
