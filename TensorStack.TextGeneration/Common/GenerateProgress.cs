// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common.Pipeline;

namespace TensorStack.TextGeneration.Common
{
    public record GenerateProgress : IRunProgress
    {
        public bool IsReset { get; set; }
        public string Result { get; set; }
    }
}