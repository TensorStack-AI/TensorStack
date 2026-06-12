// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Text.Json;
using System.Text.Json.Serialization;

namespace TensorStack.WPF
{
    public static class Serializer
    {
        private static readonly JsonSerializerOptions _cloneOptions;
        private static readonly JsonSerializerOptions _defaultOptions;

        static Serializer()
        {
            _cloneOptions = new JsonSerializerOptions
            {
                ReferenceHandler = ReferenceHandler.Preserve,
                WriteIndented = false
            };
            _defaultOptions = new JsonSerializerOptions
            {
                WriteIndented = true,
                Converters =
                {
                    new JsonStringEnumConverter()
                }
            };
        }

        public static string Serialize<T>(T data)
        {
            return JsonSerializer.Serialize(data, _defaultOptions);
        }


        public static T Deserialize<T>(string data)
        {
            return JsonSerializer.Deserialize<T>(data, _defaultOptions);
        }

        public static T DeepClone<T>(T obj)
        {
            string json = JsonSerializer.Serialize(obj, _cloneOptions);
            return JsonSerializer.Deserialize<T>(json, _cloneOptions);
        }
    }
}