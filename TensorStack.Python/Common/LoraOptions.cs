using System.Text.Json.Serialization;

namespace TensorStack.Python.Common
{
    public class LoraOptions
    {
        [JsonPropertyName("name")]
        public string Name { get; set; }

        [JsonPropertyName("strength")]
        public float Strength { get; set; }
    }
}
