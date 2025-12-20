using CSnakes.Runtime.Python;
using Microsoft.Extensions.Logging;
using System;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common.Tensor;
using TensorStack.Python.Common;

namespace TensorStack.Python
{
    public static partial class Extensions
    {
        private static readonly Regex _ansiRegex = AnsiRegex();

        /// <summary>
        /// Read JSON from file or JJSON string
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="filenameOrJson">The filename or JSON data.</param>
        public static async Task<T> ReadJsonAsync<T>(string filenameOrJson)
        {
            if (File.Exists(filenameOrJson))
            {
                using (var reader = File.OpenRead(filenameOrJson))
                {
                    return await JsonSerializer.DeserializeAsync<T>(reader);
                }
            }

            return JsonSerializer.Deserialize<T>(filenameOrJson);
        }


        /// <summary>
        /// IPyBuffer to Tensor<float>.
        /// </summary>
        /// <param name="pyBuffer">The py buffer.</param>
        public static Tensor<float> ToTensor(this IPyBuffer pyBuffer)
        {
            var buffer = pyBuffer.GetBuffer();
            var dimensions = pyBuffer.GetDimensions();
            return new Tensor<float>(buffer, dimensions);
        }


        /// <summary>
        /// Gets the IPyBuffer buffer.
        /// </summary>
        /// <param name="pyBuffer">The py buffer.</param>
        public static Memory<float> GetBuffer(this IPyBuffer pyBuffer)
        {
            return pyBuffer.AsReadOnlySpan<float>().ToArray();
        }


        /// <summary>
        /// Gets the IPyBuffer dimensions.
        /// </summary>
        /// <param name="pyBuffer">The py buffer.</param>
        public static ReadOnlySpan<int> GetDimensions(this IPyBuffer pyBuffer)
        {
            var shape = pyBuffer.Shape;
            var dimensions = new int[shape.Length];
            for (int i = 0; i < dimensions.Length; i++)
            {
                dimensions[i] = (int)shape[i];
            }
            return dimensions;
        }


        /// <summary>
        /// Extarct python logging.
        /// </summary>
        /// <typeparam name="T"></typeparam>
        /// <param name="pipelineTask">The pipeline task.</param>
        /// <param name="pythonProxy">The python proxy.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="refreshRate">The refresh rate.</param>
        /// <returns>T.</returns>
        public static async Task<T> WithPythonLogging<T>(this Task<T> pipelineTask, PipelineProxy pythonProxy, IProgress<PipelineProgress> progressCallback, int refreshRate = 250)
        {
            while (!pipelineTask.IsCompleted)
            {
                await GetPythonLogs(pythonProxy, progressCallback);
                await Task.Delay(refreshRate);
            }
            return await pipelineTask;
        }


        /// <summary>
        /// Gets the python logs.
        /// </summary>
        /// <param name="pythonProxy">The python proxy.</param>
        /// <param name="progressCallback">The progress callback.</param>
        private static async Task GetPythonLogs(PipelineProxy pythonProxy, IProgress<PipelineProgress> progressCallback)
        {
            var logResult = await pythonProxy.GetLogsAsync();
            foreach (var logEntry in logResult)
            {
                var message = _ansiRegex.Replace(logEntry.Trim([' ', '\n', '\r']), "");
                if (string.IsNullOrWhiteSpace(message))
                    continue;

                pythonProxy.Logger?.LogDebug("[PythonRuntime] {message}", message);

                var diffusionProgress = ParsePythonLog(message);
                if (message == null)
                    continue;

                progressCallback?.Report(diffusionProgress);
            }
        }
  

        /// <summary>
        /// Parses the python log.
        /// </summary>
        /// <param name="logEntry">The log entry.</param>
        /// <returns>PythonProgress.</returns>
        private static PipelineProgress ParsePythonLog(string logEntry)
        {
            try
            {
                var iteration = 0;
                var iterations = 0;
                var iterationsPerSecond = 0f;
                var secondsPerIteration = 0f;
                var megabytesSecond = 0f;
                var megabytesDownloaded = 0f;
                var megabytesTotal = 0f;
                var messageSections = logEntry.Split('|', StringSplitOptions.TrimEntries).AsSpan();
                var messageSection = messageSections[0].Split(':')[0];

                // Parse Steps / Interations
                if (messageSections.Length > 2)
                {
                    var infoSection = messageSections[2].Split('[', StringSplitOptions.TrimEntries).AsSpan();
                    var stepsSection = infoSection[0].Split('/').AsSpan();
                    if (stepsSection.Length == 2)
                    {
                        if (!int.TryParse(stepsSection[0], out iteration))
                        {
                            if (!float.TryParse(stepsSection[0].Replace('M', default), out megabytesDownloaded))
                            {
                                if (float.TryParse(stepsSection[0].Replace('G', default), out megabytesDownloaded))
                                {
                                    megabytesDownloaded *= 1000;
                                }
                            }
                        }

                        if (!int.TryParse(stepsSection[1], out iterations))
                        {
                            if (!float.TryParse(stepsSection[1].Replace('M', default), out megabytesTotal))
                            {
                                if (float.TryParse(stepsSection[1].Replace('G', default), out megabytesTotal))
                                {
                                    megabytesTotal *= 1000;
                                }
                            }
                        }
                    }

                    var iterationsSection = infoSection[1].Split(',', StringSplitOptions.TrimEntries)[1].TrimEnd(']').AsSpan();
                    if (iterationsSection.Contains("it/s".AsSpan(), StringComparison.OrdinalIgnoreCase))
                    {
                        _ = float.TryParse(iterationsSection[..^4], out iterationsPerSecond);
                        secondsPerIteration = 1f / iterationsPerSecond;
                    }
                    else if (iterationsSection.Contains("s/it".AsSpan(), StringComparison.OrdinalIgnoreCase))
                    {
                        _ = float.TryParse(iterationsSection[..^4], out secondsPerIteration);
                        iterationsPerSecond = 1f / secondsPerIteration;
                    }
                    else if (iterationsSection.Contains("MB/s".AsSpan(), StringComparison.OrdinalIgnoreCase))
                    {
                        _ = float.TryParse(iterationsSection[..^4], out megabytesSecond);
                    }
                }

                return new PipelineProgress
                {
                    Message = messageSection,
                    Iteration = iteration,
                    Iterations = iterations,
                    IterationsPerSecond = float.IsFinite(iterationsPerSecond) ? iterationsPerSecond : 0f,
                    SecondsPerIteration = float.IsFinite(secondsPerIteration) ? secondsPerIteration : 0f,
                    DownloadTotal = megabytesTotal,
                    Downloaded = megabytesDownloaded,
                    DownloadSpeed = megabytesSecond,
                    Process = "Generate"
                };

            }
            catch (Exception)
            {
                return default;
            }
        }


        /// <summary>
        /// Safely cancel
        /// </summary>
        /// <param name="cancellationTokenSource">The cancellation token source.</param>
        public static void SafeCancel(this CancellationTokenSource cancellationTokenSource)
        {
            try
            {
                cancellationTokenSource.Cancel();
            }
            catch { }
        }


        [GeneratedRegex(@"\x1B[@-_][0-?]*[ -/]*[@-~]", RegexOptions.Compiled)]
        private static partial Regex AnsiRegex();
    }
}
