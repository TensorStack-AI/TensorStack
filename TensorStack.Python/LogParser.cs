using System;
using System.Collections.Generic;
using System.Linq;
using System.Text.RegularExpressions;
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


        [GeneratedRegex(@"\x1B[@-_][0-?]*[ -/]*[@-~]", RegexOptions.Compiled)]
        private static partial Regex AnsiRegexGenerated();
    }
}
