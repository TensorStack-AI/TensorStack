using System.Text.Json.Serialization;

namespace TensorStack.Python.Common
{
    public enum QuantizationType
    {
        [JsonStringEnumMemberName("Q16Bit")]
        Q16Bit = 0,

        [JsonStringEnumMemberName("Q8Bit")]
        Q8Bit = 1,

        [JsonStringEnumMemberName("Q4Bit")]
        Q4Bit = 2
    }
}
