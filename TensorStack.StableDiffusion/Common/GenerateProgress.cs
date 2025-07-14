// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
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
        }
        public GenerateProgress(long elapsed)
        {
            Elapsed = Stopwatch.GetElapsedTime(elapsed).TotalMilliseconds;
        }
        public string Message { get; set; }

        public int BatchMax { get; set; }
        public int BatchValue { get; set; }
        public Tensor<float> BatchTensor { get; set; }

        public int StepMax { get; set; }
        public int StepValue { get; set; }
        public Tensor<float> StepTensor { get; set; }

        public double Elapsed { get; set; }
    }
}
