using System.Text.Json.Serialization;

namespace TensorStack.Python.Config
{
    public class LoraConfig
    {
        public string Path { get; set; }
        public string Name { get; set; }
        public string Weights { get; set; }

        [JsonPropertyName("is_offline_mode")]
        public bool IsOfflineMode { get; set; }
    }
}
