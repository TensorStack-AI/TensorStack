// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common;

namespace TensorStack.StableDiffusion.Config
{
    public record ResampleModelConfig : ModelConfig
    {
        public int ScaleFactor { get; set; }
    }
}
