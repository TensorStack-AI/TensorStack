using System.Collections.Generic;
using System.Text.Json.Serialization;
using TensorStack.Python.Common;

namespace TensorStack.Python.Config
{
    public sealed record PipelineConfig
    {
        [JsonPropertyName("path")]
        public string Path { get; set; }

        [JsonPropertyName("pipeline")]
        public string Pipeline { get; set; }

        [JsonPropertyName("process_type")]
        public ProcessType ProcessType { get; set; }

        [JsonPropertyName("control_net_path")]
        public string ControlNetPath { get; set; }

        [JsonPropertyName("device")]
        public string Device { get; set; } = "cuda";

        [JsonPropertyName("device_id")]
        public int DeviceId { get; set; }

        [JsonPropertyName("data_type")]
        public DataType DataType { get; set; } = DataType.Bfloat16;

        [JsonPropertyName("variant")]
        public string Variant { get; set; }

        [JsonPropertyName("cache_directory")]
        public string CacheDirectory { get; set; }

        [JsonPropertyName("secure_token")]
        public string SecureToken { get; set; }

        [JsonPropertyName("is_model_offload_enabled")]
        public bool IsModelOffloadEnabled { get; set; }

        [JsonPropertyName("is_full_offload_enabled")]
        public bool IsFullOffloadEnabled { get; set; }

        [JsonPropertyName("is_vae_slicing_enabled")]
        public bool IsVaeSlicingEnabled { get; set; }

        [JsonPropertyName("is_vae_tiling_enabled")]
        public bool IsVaeTilingEnabled { get; set; }

        [JsonPropertyName("lora_adapters")]
        public List<LoraConfig> LoraAdapters { get; set; }

    }
}
