// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common;
using TensorStack.TextGeneration.Common;

namespace TensorStack.TextGeneration.Pipelines.Florence
{
    public record FlorenceConfig : TransformerConfig
    {
        public int ImageSampleSize { get; set; } = 768;
        public int ImageSeqLength { get; set; } = 577;
        public int ImageContextWidth { get; set; } = 1000;
        public int ImageContextHeight { get; set; } = 1000;
        public ModelConfig EmbedsConfig { get; set; }
        public ModelConfig VisionConfig { get; set; }
    }
}
