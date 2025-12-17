using System.Threading.Tasks;

namespace TensorStack.Python.Config
{
    public sealed record PipelineConfig
    {
        public string Path { get; set; }
        public string Pipeline { get; set; }
        public string Device { get; set; } = "cuda";
        public int DeviceId { get; set; }
        public bool IsModelOffloadEnabled { get; set; }
        public bool IsFullOffloadEnabled { get; set; }
        public bool IsVaeSlicingEnabled { get; set; }
        public bool IsVaeTilingEnabled { get; set; }
        public DataType DataType { get; set; } = DataType.Bfloat16;
        public string Variant { get; set; }

        public static Task<PipelineConfig> FromFileAsync(string path) => Extensions.ReadJsonAsync<PipelineConfig>(path);
    }

    public enum DataType
    {
        Float16 = 0,
        Float32 = 1,
        Bfloat16 = 2,
        Float8 = 3,
        Float8_e4m3fn = 4,
        Float8_e5m2 = 5,
    }
}
