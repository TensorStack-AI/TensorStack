using System;
using System.Globalization;
using System.Text.Json.Serialization;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;

namespace TensorStack.Python.Common
{
    public record PipelineProgress : IRunProgress
    {
        public string Key { get; init; }
        public string Subkey { get; init; }
        public DateTime Timestamp { get; init; }
        public float Elapsed { get; init; }
        public int Value { get; init; }
        public int Maximum { get; init; }
        public int BatchValue { get; init; }
        public int BatchMaximum { get; init; }
        public string Message { get; init; }

        [JsonIgnore]
        public Tensor<float> Tensor { get; init; }

        public float IterationsPerSecond => Elapsed > 0 ? 1000f / Elapsed : 0;
        public float SecondsPerIteration => Elapsed > 0 ? Elapsed / 1000f : 0;

        public readonly static IProgress<PipelineProgress> ConsoleCallback = new Progress<PipelineProgress>(Console.WriteLine);


        public static PipelineProgress Create(string inputData, Tensor<float> tensor)
        {
            if (string.IsNullOrWhiteSpace(inputData))
                return null;

            // {Key}|{Subkey}|{Timestamp}|{Elapsed}|{Value}|{Maximum}|{BatchValue}|{BatchMaximum}|{Message}
            var parameters = inputData.Split('|', 9, StringSplitOptions.TrimEntries);
            if (parameters.Length < 9)
                return null;

            return new PipelineProgress
            {
                Key = parameters[0],
                Subkey = parameters[1],
                Timestamp = DateTime.Parse(parameters[2], CultureInfo.InvariantCulture),
                Elapsed = float.Parse(parameters[3], CultureInfo.InvariantCulture),
                Value = int.Parse(parameters[4], CultureInfo.InvariantCulture),
                Maximum = int.Parse(parameters[5], CultureInfo.InvariantCulture),
                BatchValue = int.Parse(parameters[6], CultureInfo.InvariantCulture),
                BatchMaximum = int.Parse(parameters[7], CultureInfo.InvariantCulture),
                Message = parameters[8],
                Tensor = tensor
            };
        }

    }
}
