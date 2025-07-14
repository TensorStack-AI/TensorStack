// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common;

namespace TensorStack.StableDiffusion.Config
{
    public record ControlNetModelConfig : ModelConfig
    {
        public string Name { get; set; }
        public bool InvertInput { get; set; }
        public int LayerCount { get; set; }
        public bool DisablePooledProjection { get; set; }
    }
}
