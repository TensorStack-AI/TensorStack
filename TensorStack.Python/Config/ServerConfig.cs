using System.Text.Json.Serialization;
using System.Threading.Tasks;

namespace TensorStack.Python.Config
{
    public record ServerConfig
    {
        public const string Name = "TensorStack.Python.Server";
        public const string ObjectPipeName = "TensorStack.Python.Object";
        public const string MessagePipeName = "TensorStack.Python.Message";

        public bool IsDebug { get; set; }
        public string Environment { get; set; }
        public string[] Requirements { get; set; }
        public string Directory { get; set; }
        public static Task<ServerConfig> FromFileAsync(string path) => Extensions.ReadJsonAsync<ServerConfig>(path);
    }

    public record ClientConfig : ServerConfig
    {
        public ClientConfig() { }
        public ClientConfig(ServerConfig config) : base(config) { }

        [JsonIgnore]
        public string Path { get; set; }
    }
}
