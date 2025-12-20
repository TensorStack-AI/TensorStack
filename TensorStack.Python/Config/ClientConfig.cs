using System.Text.Json.Serialization;

namespace TensorStack.Python.Config
{
    public record ClientConfig : ServerConfig
    {
        public ClientConfig() { }
        public ClientConfig(ServerConfig config) : base(config) { }

        [JsonIgnore]
        public string Path { get; set; }
    }
}
