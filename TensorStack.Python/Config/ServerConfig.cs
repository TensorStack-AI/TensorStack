using System.Threading.Tasks;

namespace TensorStack.Python.Config
{
    public record ServerConfig
    {
        public const string Name = "Python.Server";
        public const string Executable = "Python.Server.exe";
        public const string ObjectPipeName = "TensorStack.Python.Object";
        public const string MessagePipeName = "TensorStack.Python.Message";

        public bool IsDebug { get; set; }
        public string Environment { get; set; }
        public string[] Requirements { get; set; }
        public string Directory { get; set; }

        public static Task<ServerConfig> FromFileAsync(string path) => Extensions.ReadJsonAsync<ServerConfig>(path);
    }
}
