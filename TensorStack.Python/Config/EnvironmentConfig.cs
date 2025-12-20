using System.Threading.Tasks;

namespace TensorStack.Python.Config
{
    public record EnvironmentConfig
    {
        public bool IsDebug { get; set; }
        public string Environment { get; set; }
        public string[] Requirements { get; set; }
        public string Directory { get; set; }

        public static Task<EnvironmentConfig> FromFileAsync(string path) => Extensions.ReadJsonAsync<EnvironmentConfig>(path);
    }
}
