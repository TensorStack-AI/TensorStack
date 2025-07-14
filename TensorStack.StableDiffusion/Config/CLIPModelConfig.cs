// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common;

namespace TensorStack.StableDiffusion.Config
{
    public record CLIPModelConfig : ModelConfig
    {
        public int HiddenSize { get; set; } = 768;
        public int SequenceLength { get; set; } = 77;
        public int PadTokenId { get; set; } = 1;
        public bool IsFixedSequenceLength { get; set; } = true;
    }
}
