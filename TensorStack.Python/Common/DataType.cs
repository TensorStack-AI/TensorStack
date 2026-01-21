using System.Text.Json.Serialization;

namespace TensorStack.Python.Common
{
    public enum DataType
    {
        [JsonStringEnumMemberName("float32")]
        Float32 = 0,

        [JsonStringEnumMemberName("bfloat16")]
        Bfloat16 = 1,

        [JsonStringEnumMemberName("float16")]
        Float16 = 2,

        [JsonStringEnumMemberName("float8")]
        Float8 = 3,

        [JsonStringEnumMemberName("int8")]
        Int8 = 6
    }
}
