using System;
using System.Collections.Generic;
using System.Globalization;

namespace TensorStack.Python
{

    internal record LogEntry(DateTime Timestamp, string Message);

    internal static class LogParser
    {
        /// <summary>
        /// Parses the python logs.
        /// </summary>
        /// <param name="logEntries">The log entries.</param>
        /// <returns>IEnumerable&lt;PipelineProgress&gt;.</returns>
        internal static IEnumerable<LogEntry> ParseLogs(IReadOnlyList<string> logEntries)
        {
            foreach (var logEntry in logEntries)
            {
                var progress = ParsePythonLog(logEntry);
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
        private static LogEntry ParsePythonLog(string logEntry)
        {
            try
            {
                var messageSections = logEntry.Split('|', 3, StringSplitOptions.TrimEntries).AsSpan();
                if (messageSections.Length < 2)
                    return default;

                var message = messageSections[1].Trim([' ', '\n', '\r']);
                if (message.Length < 5)
                    return default;

                return new LogEntry(DateTime.Parse(messageSections[0], CultureInfo.InvariantCulture), message);
            }
            catch (Exception)
            {
                return default;
            }
        }
    }
}
