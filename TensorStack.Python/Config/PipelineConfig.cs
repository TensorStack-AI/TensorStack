using System.Collections.Generic;
using TensorStack.Python.Common;

namespace TensorStack.Python.Config
{
    public sealed record PipelineConfig
    {
        public string Path { get; set; }
        public string Pipeline { get; set; }
        public ProcessType ProcessType { get; set; }
        public string Device { get; set; } = "cuda";
        public int DeviceId { get; set; }
        public bool IsModelOffloadEnabled { get; set; }
        public bool IsFullOffloadEnabled { get; set; }
        public bool IsVaeSlicingEnabled { get; set; }
        public bool IsVaeTilingEnabled { get; set; }
        public DataType DataType { get; set; } = DataType.Bfloat16;
        public string Variant { get; set; }
        public string CacheDirectory { get; set; }
        public string SecureToken { get; set; }
        public List<LoraConfig> LoraAdapters { get; set; }
    }
}
