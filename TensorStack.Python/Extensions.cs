using CSnakes.Runtime.Python;
using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.IO;
using System.IO.Pipes;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.Json;
using System.Text.RegularExpressions;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common.Tensor;
using TensorStack.Python.Options;

namespace TensorStack.Python
{
    public static class Extensions
    {
        private static readonly Regex _ansiRegex = new Regex(@"\x1B[@-_][0-?]*[ -/]*[@-~]", RegexOptions.Compiled);

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
        /// Sends a PythonMessage.
        /// </summary>
        /// <typeparam name="T">IPythonMessage</typeparam>
        /// <param name="pipe">The pipe.</param>
        /// <param name="message">The message.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        public static async Task SendMessage<T>(this PipeStream pipe, T message, CancellationToken cancellationToken = default) where T : IPythonMessage
        {
            var intBuffer = new byte[4];
            var tensors = message.Tensors ?? [];
            var jsonData = JsonSerializer.Serialize(message);
            var jsonBytes = Encoding.UTF8.GetBytes(jsonData);

            // Write tensor count
            BitConverter.TryWriteBytes(intBuffer, tensors.Count);
            await pipe.WriteAsync(intBuffer, cancellationToken);

            // Write JSON length
            BitConverter.TryWriteBytes(intBuffer, jsonBytes.Length);
            await pipe.WriteAsync(intBuffer, cancellationToken);

            // Tensors
            foreach (var tensor in tensors)
            {
                // Rank
                BitConverter.TryWriteBytes(intBuffer, tensor.Rank);
                await pipe.WriteAsync(intBuffer, cancellationToken);

                // Dimensions
                foreach (var dim in tensor.Dimensions.ToArray())
                {
                    BitConverter.TryWriteBytes(intBuffer, dim);
                    await pipe.WriteAsync(intBuffer, cancellationToken);
                }

                // Tensor buffer
                await pipe.WriteAsync(new byte[] { 0 }, cancellationToken); // float32
                var tensorBytes = MemoryMarshal.AsBytes(tensor.Memory.Span).ToArray();
                BitConverter.TryWriteBytes(intBuffer, tensorBytes.Length);
                await pipe.WriteAsync(intBuffer, cancellationToken);
                await pipe.WriteAsync(tensorBytes, cancellationToken);
            }

            // JSON
            await pipe.WriteAsync(jsonBytes, cancellationToken);
            await pipe.FlushAsync(cancellationToken);
        }


        /// <summary>
        /// Receives a PythonMessage message.
        /// </summary>
        /// <typeparam name="T">IPythonMessage</typeparam>
        /// <param name="pipe">The pipe.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        public static async Task<T> ReceiveMessage<T>(this PipeStream pipe, CancellationToken cancellationToken = default) where T : IPythonMessage
        {
            var tensorCountBytes = await pipe.ReadExactlyAsync(4, cancellationToken);
            int tensorCount = BitConverter.ToInt32(tensorCountBytes);
            var jsonLengthBytes = await pipe.ReadExactlyAsync(4, cancellationToken);
            int jsonLength = BitConverter.ToInt32(jsonLengthBytes);

            // Tensors
            var tensors = new List<Tensor<float>>();
            for (int t = 0; t < tensorCount; t++)
            {
                // Rank
                var rankBytes = await pipe.ReadExactlyAsync(4, cancellationToken);
                int rank = BitConverter.ToInt32(rankBytes);

                // Dimensions
                int[] dims = new int[rank];
                for (int d = 0; d < rank; d++)
                {
                    var dimBytes = await pipe.ReadExactlyAsync(4, cancellationToken);
                    dims[d] = BitConverter.ToInt32(dimBytes);
                }

                // Type byte
                var typeByte = await pipe.ReadExactlyAsync(1, cancellationToken);
                if (typeByte[0] != 0)
                    throw new NotSupportedException("Only float32 tensors are supported.");

                // Tensor buffer length
                var bufferLenBytes = await pipe.ReadExactlyAsync(4, cancellationToken);
                int bufferLen = BitConverter.ToInt32(bufferLenBytes);

                // Tensor buffer
                var floats = new float[bufferLen / 4];
                var tensorBytes = await pipe.ReadExactlyAsync(bufferLen, cancellationToken);
                Buffer.BlockCopy(tensorBytes, 0, floats, 0, bufferLen);

                var tensor = new Tensor<float>(dims);
                floats.CopyTo(tensor.Memory.Span);
                tensors.Add(tensor);
            }

            // JSON
            var jsonBytes = await pipe.ReadExactlyAsync(jsonLength, cancellationToken);
            string json = Encoding.UTF8.GetString(jsonBytes);
            var response = JsonSerializer.Deserialize<T>(json);
            response.Tensors = tensors;
            return response;
        }


        /// <summary>
        /// Sends a object as JSON.
        /// </summary>
        /// <typeparam name="T">The object type</typeparam>
        /// <param name="pipe">The pipe.</param>
        /// <param name="dataObject">The object to send.</param>
        public static async Task SendObject<T>(this PipeStream pipe, T dataObject)
        {
            var json = JsonSerializer.Serialize(dataObject);
            var jsonBytes = Encoding.UTF8.GetBytes(json);
            var lengthBytes = BitConverter.GetBytes(jsonBytes.Length);
            await pipe.WriteAsync(lengthBytes, 0, lengthBytes.Length);
            await pipe.WriteAsync(jsonBytes, 0, jsonBytes.Length);
            await pipe.FlushAsync();
        }


        /// <summary>
        /// Receives the object as JSON.
        /// </summary>
        /// <typeparam name="T">The object type</typeparam>
        /// <param name="pipe">The pipe.</param>
        public static async Task<T> ReceiveObject<T>(this PipeStream pipe)
        {
            var lengthBytes = new byte[4];
            await pipe.ReadExactlyAsync(lengthBytes, 0, lengthBytes.Length);
            int jsonLength = BitConverter.ToInt32(lengthBytes);

            byte[] jsonData = new byte[jsonLength];
            await pipe.ReadExactlyAsync(jsonData, 0, jsonLength);

            string jsonString = Encoding.UTF8.GetString(jsonData);
            return JsonSerializer.Deserialize<T>(jsonString);
        }


        public static Task SendResponse(this PipeStream pipe, CancellationToken cancellationToken = default)
        {
            return pipe.SendMessage(new PythonResponseMessage { Tensors = [] }, cancellationToken);
        }

        /// <summary>
        /// Read exactly n bytes
        /// </summary>
        /// <param name="stream">The stream.</param>
        /// <param name="count">The count.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public static async Task<byte[]> ReadExactlyAsync(this Stream stream, int count, CancellationToken cancellationToken = default)
        {
            int offset = 0;
            var buffer = new byte[count];
            while (offset < count)
            {
                int read = await stream.ReadAsync(buffer.AsMemory(offset, count - offset), cancellationToken);
                if (read == 0)
                    throw new EndOfStreamException();
                offset += read;
            }
            return buffer;
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
        public static async Task<T> WithPythonLogging<T>(this Task<T> pipelineTask, PythonProxy pythonProxy, IProgress<PythonProgress> progressCallback, int refreshRate = 250)
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
        private static async Task GetPythonLogs(PythonProxy pythonProxy, IProgress<PythonProgress> progressCallback)
        {
            var logResult = await pythonProxy.GetLogsAsync();
            foreach (var logEntry in logResult)
            {
                var message = _ansiRegex.Replace(logEntry.Trim([' ', '\n', '\r']), "");
                if (string.IsNullOrWhiteSpace(message))
                    continue;

                pythonProxy.Logger.LogDebug(message);

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
        private static PythonProgress ParsePythonLog(string logEntry)
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
                                    megabytesDownloaded = megabytesDownloaded * 1000;
                                }
                            }
                        }

                        if (!int.TryParse(stepsSection[1], out iterations))
                        {
                            if (!float.TryParse(stepsSection[1].Replace('M', default), out megabytesTotal))
                            {
                                if (float.TryParse(stepsSection[1].Replace('G', default), out megabytesTotal))
                                {
                                    megabytesTotal = megabytesTotal * 1000;
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

                return new PythonProgress
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

    }
}
