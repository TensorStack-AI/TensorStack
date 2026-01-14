using System.Text.Json.Serialization;

namespace TensorStack.Python.Common
{
    public enum DataType
    {
        [JsonStringEnumMemberName("float16")]
        Float16 = 0,

        [JsonStringEnumMemberName("float32")]
        Float32 = 1,

        [JsonStringEnumMemberName("bfloat16")]
        Bfloat16 = 2,

        [JsonStringEnumMemberName("float8_e4m3fn")]
        Float8_e4m3fn = 4,

        [JsonStringEnumMemberName("float8_e5m2")]
        Float8_e5m2 = 5,
    }
}
