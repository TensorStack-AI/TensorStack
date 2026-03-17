using System.Text.Json.Serialization;

namespace TensorStack.Python.Common
{
    public enum MemoryModeType
    {
        /// <summary>
        /// Device Offload
        /// </summary>
        [JsonStringEnumMemberName("OffloadGPU")]
        Device = 0,

        /// <summary>
        /// Sequential CPU Offload
        /// </summary>
        [JsonStringEnumMemberName("OffloadCPU")]
        OffloadCPU = 1,

        /// <summary>
        /// Model CPU Offload
        /// </summary>
        [JsonStringEnumMemberName("OffloadModel")]
        OffloadModel = 2,

        /// <summary>
        /// Hardware Balanced
        /// </summary>
        [JsonStringEnumMemberName("Balanced")]
        Balanced = 3 
    }
}
