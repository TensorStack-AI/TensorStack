// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;
using TensorStack.Common.Vision;

namespace TensorStack.Florence.Common
{
    public record GenerateOptions : IRunOptions
    {
        public TaskType TaskType { get; set; }
        public string Prompt { get; set; }
        public ImageTensor Image { get; set; }
        public CoordinateBox<float> Region { get; set; }

        public int TopK { get; set; } = 50;
        public int NumBeams { get; set; } = 1;
        public int MaxLength { get; set; } = 1024;
        public int NoRepeatNgramSize { get; set; } = 3;
    }
}
