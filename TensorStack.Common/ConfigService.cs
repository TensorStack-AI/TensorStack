// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.IO;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace TensorStack.Common
{
    public class ConfigService
    {
              /// <summary>
        /// Serializes the specified configuration to file.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="configuration">The configuration.</param>
        /// <param name="useRelativePaths">if set to <c>true</c> use relative paths.</param>
        public static void Serialize<T>(string configFile, T configuration, bool useRelativePaths = true)
        {
            var serializerOptions = GetSerializerOptions(configFile, useRelativePaths);
            using (var configFileStream = File.Open(configFile, FileMode.Create))
            {
                JsonSerializer.Serialize<T>(configFileStream, configuration, serializerOptions);
            }
        }


        /// <summary>
        /// Deserializes the specified configuration file.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="useRelativePaths">if set to <c>true</c> [use relative paths].</param>
        /// <returns>T.</returns>
        public static T Deserialize<T>(string configFile, bool useRelativePaths = true)
        {
            var serializerOptions = GetSerializerOptions(configFile, useRelativePaths);
            using (var configFileStream = File.OpenRead(configFile))
            {
                return JsonSerializer.Deserialize<T>(configFileStream, serializerOptions);
            }
        }


        /// <summary>
        /// Gets the serializer options.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        /// <param name="useRelativePaths">if set to <c>true</c> use relative paths.</param>
        private static JsonSerializerOptions GetSerializerOptions(string filePath, bool useRelativePaths = true)
        {
            var serializerOptions = new JsonSerializerOptions
            {
                WriteIndented = true,
                Converters = { new JsonStringEnumConverter() }
            };

            if (useRelativePaths)
                serializerOptions.Converters.Add(new RelativePathConverter(Path.GetDirectoryName(Path.GetFullPath(filePath))));

            return serializerOptions;
        }
    }

    internal class RelativePathConverter : JsonConverter<string>
    {
        private readonly string _baseDir;

        /// <summary>
        /// Initializes a new instance of the <see cref="RelativePathConverter"/> class.
        /// </summary>
        /// <param name="baseDir">The base dir.</param>
        public RelativePathConverter(string baseDir)
        {
            _baseDir = baseDir;
        }

        /// <summary>
        /// Replace relative paths with full path.
        /// </summary>
        /// <param name="reader">The reader.</param>
        /// <param name="typeToConvert">The type to convert.</param>
        /// <param name="options">An object that specifies serialization options to use.</param>
        /// <returns>The converted value.</returns>
        public override string Read(ref Utf8JsonReader reader, Type typeToConvert, JsonSerializerOptions options)
        {
            var value = reader.GetString();
            if (string.IsNullOrEmpty(value))
                return value;

            if (!Path.IsPathRooted(value) && Path.HasExtension(value))
                value = Path.Combine(_baseDir, value);

            return value;
        }


        /// <summary>
        /// Replace full path with relative path.
        /// </summary>
        /// <param name="writer">The writer to write to.</param>
        /// <param name="value">The value to convert to JSON.</param>
        /// <param name="options">An object that specifies serialization options to use.</param>
        public override void Write(Utf8JsonWriter writer, string value, JsonSerializerOptions options)
        {
            if (Path.IsPathRooted(value))
                value = Path.GetRelativePath(_baseDir, value);

            writer.WriteStringValue(value);
        }
    }
}
