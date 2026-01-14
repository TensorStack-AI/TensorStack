using System.Collections.Generic;
using System.Linq;
using System.Text.Json;
using System.Text.Json.Serialization;
using TensorStack.Common;

namespace TensorStack.Python
{
    public static class PythonSerializer
    {
        private static JsonSerializerOptions _serializerOptions;

        static PythonSerializer()
        {
            _serializerOptions = new JsonSerializerOptions
            {
                PropertyNamingPolicy = JsonNamingPolicy.SnakeCaseLower,
                NumberHandling = JsonNumberHandling.AllowReadingFromString,
                DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
                Converters = { new JsonStringEnumConverter() }
            };
        }


        public static Dictionary<string, object> ToPythonDictionary<T>(this T source, params string[] ignoreProperties) where T : class
        {
            var json = JsonSerializer.Serialize<T>(source, _serializerOptions);
            var dict = JsonSerializer.Deserialize<Dictionary<string, object>>(json, _serializerOptions);
            return dict.ToJsonElementDictionary(ignoreProperties);
        }


        private static Dictionary<string, object> ToJsonElementDictionary(this Dictionary<string, object> source, params string[] ignoreProperties)
        {
            var result = new Dictionary<string, object>();
            foreach (var (key, value) in source)
            {
                if (ignoreProperties.Contains(key))
                    continue;

                result[key] = ConvertValue(value);
            }

            return result;
        }


        private static object ConvertValue(object value)
        {
            if (value is not JsonElement el)
                return value;

            return el.ValueKind switch
            {
                JsonValueKind.Number =>
                    el.TryGetInt64(out var l) ? l :
                    el.TryGetDouble(out var d) ? d :
                    null,

                JsonValueKind.String =>
                    el.GetString(),

                JsonValueKind.True => true,
                JsonValueKind.False => false,

                JsonValueKind.Array =>
                    el.EnumerateArray()
                      .Select(ConvertElement)
                      .ToArray(),

                JsonValueKind.Object =>
                    el.EnumerateObject()
                      .ToDictionary(p => p.Name, p => ConvertElement(p.Value)),

                JsonValueKind.Null => null,

                _ => null
            };
        }

        private static object ConvertElement(JsonElement el) => ConvertValue(el);
    }
}

