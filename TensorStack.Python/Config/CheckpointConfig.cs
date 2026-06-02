using System.Text.Json.Serialization;

namespace TensorStack.Python.Config
{
    public sealed record CheckpointConfig
    {
        [JsonPropertyName("text_encoder")]
        public string TextEncoder { get; set; }

        [JsonPropertyName("text_encoder_2")]
        public string TextEncoder2 { get; set; }

        [JsonPropertyName("text_encoder_3")]
        public string TextEncoder3 { get; set; }

        [JsonPropertyName("unet")]
        public string Unet { get; set; }

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

        [JsonPropertyName("latent_upsampler")]
        public string LatentUpsampler { get; set; }

        [JsonPropertyName("latent_upsampler_temporal")]
        public string LatentUpsamplerTemporal { get; set; }

        [JsonPropertyName("condition_encoder")]
        public string ConditionEncoder { get; set; }

        [JsonPropertyName("audio_tokenizer")]
        public string AudioTokenizer { get; set; }

        [JsonPropertyName("audio_token_detokenizer")]
        public string AudioDetokenizer { get; set; }
    }
}
