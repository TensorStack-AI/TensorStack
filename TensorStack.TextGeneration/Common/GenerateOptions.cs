// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common.Pipeline;

namespace TensorStack.TextGeneration.Common
{
    public record GenerateOptions : IRunOptions
    {
        public string Prompt { get; set; }
        public int MinLength { get; set; } = 20;
        public int MaxLength { get; set; } = 200;
        public int NoRepeatNgramSize { get; set; } = 3;

        public int Seed { get; set; }
        public int Beams { get; set; } = 1;
        public int TopK { get; set; } = 1;
        public float TopP { get; set; } = 0.9f;
        public float Temperature { get; set; } = 1.0f;
        public float LengthPenalty { get; set; } = 1.0f;
        public EarlyStopping EarlyStopping { get; set; }
        public int DiversityLength { get; set; } = 20;
    }
}
