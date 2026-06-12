using System.Collections.Generic;

namespace Amuse.Common.Config
{
    public sealed record ServerConfig
    {
        public int ChunkSize { get; } = 32 * 1024 * 1024; // 32 MB
        public string Name { get; init; }
        public string Executable { get; init; }
        public string[] Arguments { get; set; }
        public string ChannelCommand { get; init; }
        public string ChannelPipeName { get; init; }
        public string ChannelProgress { get; init; }
        public string DirectoryBase { get; init; }

        public static ServerConfig GetConfig(ServerType serverType, string directoryBase = null)
        {
            return _configurations[serverType] with
            {
                DirectoryBase = directoryBase,
            };
        }


        private readonly static Dictionary<ServerType, ServerConfig> _configurations = new Dictionary<ServerType, ServerConfig>
        {
            {
                ServerType.OnnxRuntime,  new ServerConfig
                {
                    Name = "AmuseOnnx",
                    Arguments = [nameof(ServerType.OnnxRuntime)],
                    Executable = "AmuseHost.Onnx.exe",
                    ChannelCommand = "AmuseOnnx.Command",
                    ChannelPipeName = "AmuseOnnx.PipeName",
                    ChannelProgress = "AmuseOnnx.Progress"
                }
            },
            {
                ServerType.PyTorch,  new ServerConfig
                {
                    Name = "AmusePyTorch",
                    Arguments = [nameof(ServerType.PyTorch)],
                    Executable = "AmuseHost.PyTorch.exe",
                    ChannelCommand = "AmusePyTorch.Command",
                    ChannelPipeName = "AmusePyTorch.PipeName",
                    ChannelProgress = "AmusePyTorch.Progress"
                }
            }
        };
    }
}
