// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Diagnostics;

namespace TensorStack.Common.Pipeline
{
    /// <summary>
    /// RunProgress calss for reporting Pipeline progress.
    /// </summary>
    public record RunProgress
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="RunProgress"/> class.
        /// </summary>
        /// <param name="message">The message.</param>
        /// <param name="elapsed">The timestamp.</param>
        public RunProgress(string message, long timestamp)
        {
            Message = message;
            Elapsed = Stopwatch.GetElapsedTime(timestamp);
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="RunProgress"/> class.
        /// </summary>
        /// <param name="value">The value.</param>
        /// <param name="maximum">The maximum.</param>
        /// <param name="timestamp">The timestamp.</param>
        /// <param name="message">The message.</param>
        public RunProgress(int value, int maximum, long timestamp = default, string message = default)
            : this(message, timestamp)
        {
            Value = value;
            Maximum = maximum;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="RunProgress"/> class.
        /// </summary>
        /// <param name="message">The message.</param>
        public RunProgress(string message)
            : this(message, default) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="RunProgress"/> class.
        /// </summary>
        /// <param name="timestamp">The timestamp.</param>
        public RunProgress(long timestamp)
            : this(default, timestamp) { }

        /// <summary>
        /// Gets the progress value.
        /// </summary>
        public int Value { get; }

        /// <summary>
        /// Gets the progress maximum.
        /// </summary>
        public int Maximum { get; }

        /// <summary>
        /// Gets the message.
        /// </summary>
        public string Message { get; }

        /// <summary>
        /// Gets or elapsed time.
        /// </summary>
        public TimeSpan Elapsed { get; }

        /// <summary>
        /// Gets the get timestamp.
        /// </summary>
        public static long GetTimestamp() => Stopwatch.GetTimestamp();
    }
}
