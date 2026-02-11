using System.Collections.Generic;
using System.Text.Json.Serialization;
using TensorStack.Python.Config;

namespace TensorStack.Python.Common
{
    public sealed record PipelineReloadOptions
    {
        [JsonPropertyName("process_type")]
        public ProcessType ProcessType { get; set; }

        [JsonPropertyName("control_net")]
        public ControlNetConfig ControlNet { get; set; }

        [JsonPropertyName("lora_adapters")]
        public List<LoraConfig> LoraAdapters { get; set; }
    }
}
