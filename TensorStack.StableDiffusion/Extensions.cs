// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.Extensions.Logging;
using System;
using System.Diagnostics;
using TensorStack.StableDiffusion.Common;

namespace TensorStack.StableDiffusion
{
    public static class Extensions
    {
        /// <summary>
        /// Notifies the specified message.
        /// </summary>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="message">The message.</param>
        public static void Notify(this IProgress<GenerateProgress> progressCallback, string message)
        {
            progressCallback?.Report(new GenerateProgress(message));
        }


        /// <summary>
        /// Log and return timestamp.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="level">The level.</param>
        /// <param name="message">The message.</param>
        /// <param name="parameters">The parameters.</param>
        public static long LogBegin(this ILogger logger, LogLevel level, string message, params object[] parameters)
        {
            logger?.Log(level, message, parameters);
            return Stopwatch.GetTimestamp();
        }


        /// <summary>
        /// Logs the end of scope with begin timestamp.
        /// </summary>
        /// <param name="logger">The logger.</param>
        /// <param name="level">The level.</param>
        /// <param name="timestamp">The timestamp.</param>
        /// <param name="message">The message.</param>
        /// <param name="parameters">The parameters.</param>
        public static void LogEnd(this ILogger logger, LogLevel level, long timestamp, string message, params object[] parameters)
        {
            var elapsed = Stopwatch.GetElapsedTime(timestamp);
            var formatted = string.Format(message, parameters);
            logger?.Log(level, "{formatted}, Elapsed: {elapsed}", formatted, elapsed);
        }
    }
}
