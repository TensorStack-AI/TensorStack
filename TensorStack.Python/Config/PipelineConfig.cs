using System.Collections.Generic;
using System.Text.Json.Serialization;
using TensorStack.Python.Common;

namespace TensorStack.Python.Config
{
    public sealed record PipelineConfig
    {
        [JsonPropertyName("base_model_path")]
        public string BaseModelPath { get; set; }

        [JsonPropertyName("pipeline")]
        public string Pipeline { get; set; }

        [JsonPropertyName("process_type")]
        public ProcessType ProcessType { get; set; }

        [JsonPropertyName("device")]
        public string Device { get; set; } = "cuda";

        [JsonPropertyName("device_id")]
        public int DeviceId { get; set; }

        [JsonPropertyName("data_type")]
        public DataType DataType { get; set; } = DataType.Bfloat16;

        [JsonPropertyName("quant_data_type")]
        public DataType QuantDataType { get; set; } = DataType.Bfloat16;

        [JsonPropertyName("variant")]
        public string Variant { get; set; }

        [JsonPropertyName("cache_directory")]
        public string CacheDirectory { get; set; }

        [JsonPropertyName("secure_token")]
        public string SecureToken { get; set; }

        [JsonPropertyName("lora_adapters")]
        public List<LoraConfig> LoraAdapters { get; set; }

        [JsonPropertyName("control_net")]
        public ControlNetConfig ControlNet { get; set; }

        [JsonPropertyName("memory_mode")]
        public MemoryModeType MemoryMode { get; set; }

        [JsonPropertyName("checkpoint_config")]
        public CheckpointConfig CheckpointConfig { get; set; }
    }
}
