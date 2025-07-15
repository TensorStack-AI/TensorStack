// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Diagnostics;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;

namespace TensorStack.StableDiffusion.Common
{
    public record GenerateProgress : IRunProgress
    {
        public GenerateProgress() { }
        public GenerateProgress(string message)
        {
            Message = message;
            Type = ProgressType.Message;
        }
        public GenerateProgress(long elapsed)
        {
            Elapsed = Stopwatch.GetElapsedTime(elapsed);
        }
        public ProgressType Type { get; set; }
        public string Message { get; set; }
        public TimeSpan Elapsed { get; set; }

        public int Max { get; set; }
        public int Value { get; set; }
        public Tensor<float> Tensor { get; set; }

        public enum ProgressType
        {
            Message = 0,
            Step = 1
        }
    }
}
