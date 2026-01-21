using Microsoft.Extensions.Logging;
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using System.Text.RegularExpressions;
using TensorStack.Common;
using TensorStack.Python.Common;

namespace TensorStack.Python
{
    internal static partial class LogParser
    {
        private static readonly Regex AnsiRegex = AnsiRegexGenerated();


        /// <summary>
        /// Parses the python logs.
        /// </summary>
        /// <param name="logEntries">The log entries.</param>
        /// <returns>IEnumerable&lt;PipelineProgress&gt;.</returns>
        internal static IEnumerable<PipelineProgress> ParseLogs(IReadOnlyList<string> logEntries)
        {
            foreach (var logEntry in logEntries)
            {
                var message = AnsiRegex.Replace(logEntry.Trim([' ', '\n', '\r']), "");
                if (string.IsNullOrWhiteSpace(message))
                    continue;

                var progress = ParsePythonLog(message);
                if (progress == null)
                    continue;

                yield return progress;
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
                var messageSections = logEntry.Split('|', StringSplitOptions.TrimEntries).AsSpan();
                if (messageSections.Length > 0)
                {
                    if (messageSections[0].Equals("[HUB_DOWNLOAD]"))
                    {
                        if (messageSections.Length >= 8)
                        {
                            var modelName = messageSections[1];
                            var modelFileName = messageSections[2];
                            var modelProgress = int.Parse(messageSections[3]);
                            var modelTotal = int.Parse(messageSections[4]);
                            var progress = int.Parse(messageSections[5]);
                            var progressTotal = int.Parse(messageSections[6]);
                            var speed = float.Parse(messageSections[7]);

                            return new PipelineProgress
                            {
                                Process = "Download",
                                Message = logEntry.Replace("[HUB_DOWNLOAD]", "[Download]"),
                                Iteration = progress,
                                Iterations = progressTotal,
                                DownloadModel = modelName,
                                DownloadFile = modelFileName,
                                //DownloadTotal = megabytesTotal,
                                //Downloaded = megabytesDownloaded,
                                DownloadSpeed = speed,
                            };
                        }
                    }
                    else
                    {
                        // Parse Steps / Interations
                        if (messageSections.Length > 2)
                        {
                            var iteration = 0;
                            var iterations = 0;
                            var iterationsPerSecond = 0f;
                            var secondsPerIteration = 0f;
                            var messageSection = messageSections[0].Split(':')[0];
                            var infoSection = messageSections[2].Split('[', StringSplitOptions.TrimEntries).AsSpan();
                            var stepsSection = infoSection[0].Split('/').AsSpan();
                            if (stepsSection.Length == 2)
                            {
                                _ = int.TryParse(stepsSection[0], out iteration);
                                _ = int.TryParse(stepsSection[1], out iterations);
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

                            return new PipelineProgress
                            {
                                Message = messageSection,
                                Iteration = iteration,
                                Iterations = iterations,
                                IterationsPerSecond = float.IsFinite(iterationsPerSecond) ? iterationsPerSecond : 0f,
                                SecondsPerIteration = float.IsFinite(secondsPerIteration) ? secondsPerIteration : 0f,
                                Process = messageSection.StartsWith("Loading") ? "Load" : "Generate"
                            };
                        }
                    }
                }

                return new PipelineProgress
                {
                    Message = logEntry
                };
            }
            catch (Exception)
            {
                return default;
            }
        }


        [GeneratedRegex(@"\x1B[@-_][0-?]*[ -/]*[@-~]", RegexOptions.Compiled)]
        private static partial Regex AnsiRegexGenerated();
    }
}
