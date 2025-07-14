// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common;

namespace TensorStack.StableDiffusion.Config
{
    public record PaellaVQModelConfig : ModelConfig
    {
        public int Scale { get; set; } = 4;
        public float ScaleFactor { get; set; }
        public float ShiftFactor { get; set; }
        public int OutChannels { get; set; } = 3;
    }
}
