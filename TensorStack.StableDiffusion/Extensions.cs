// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.Extensions.Logging;
using System;
using System.Diagnostics;
using TensorStack.Common.Tensor;
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
        /// Notifies the specified message.
        /// </summary>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="message">The message.</param>
        public static void Notify(this IProgress<GenerateProgress> progressCallback, GenerateProgress message)
        {
            progressCallback?.Report(message);
        }


        /// <summary>
        /// Notifies the specified step.
        /// </summary>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="step">The step.</param>
        /// <param name="steps">The steps.</param>
        /// <param name="latents">The latents.</param>
        /// <param name="elapsed">The elapsed.</param>
        public static void Notify(this IProgress<GenerateProgress> progressCallback, int step, int steps, Tensor<float> latents, long elapsed)
        {
            progressCallback?.Report(new GenerateProgress
            {
                Max = steps,
                Value = step,
                Tensor = latents.Clone(),
                Type = GenerateProgress.ProgressType.Step,
                Message = $"Step: {step:D2}/{steps:D2}",
                Elapsed = elapsed > 0 ? Stopwatch.GetElapsedTime(elapsed) : TimeSpan.Zero
            });
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
