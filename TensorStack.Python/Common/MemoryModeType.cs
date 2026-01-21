using System.Text.Json.Serialization;

namespace TensorStack.Python.Common
{
    public enum MemoryModeType
    {
        /// <summary>
        /// Selected Device
        /// </summary>
        [JsonStringEnumMemberName("Device")]
        Device = 0, // single device

        /// <summary>
        /// Muti-Device (Automatic distribution)
        /// </summary>
        [JsonStringEnumMemberName("MultiDevice")]
        MultiDevice = 2,

        /// <summary>
        /// Sequential CPU Offload
        /// </summary>
        [JsonStringEnumMemberName("OffloadCPU")]
        OffloadCPU = 3,

        /// <summary>
        /// Model CPU Offload
        /// </summary>
        [JsonStringEnumMemberName("OffloadModel")]
        OffloadModel = 4,


        /// <summary>
        /// Device + VAE Slice and Tile
        /// </summary>
        [JsonStringEnumMemberName("LowMemDevice")]
        LowMemDevice = 10,

        /// <summary>
        /// Model CPU Offload + VAE Slice and Tile
        /// </summary>
        [JsonStringEnumMemberName("LowMemOffloadModel")]
        LowMemOffloadModel = 14
    }
}
