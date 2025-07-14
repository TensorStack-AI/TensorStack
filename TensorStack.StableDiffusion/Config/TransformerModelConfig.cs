// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Text.Json.Serialization;
using TensorStack.Common;
using TensorStack.StableDiffusion.Enums;

namespace TensorStack.StableDiffusion.Config
{
    public record TransformerModelConfig :ModelConfig
    {
        public int InChannels { get; set; } = 16;
        public int OutChannels { get; set; } = 16;
        public int JointAttention { get; set; } = 4096;
        public int PooledProjection { get; set; } = 2048;
        public int CaptionProjection { get; set; } = 1536;

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public ModelType ModelType { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public string ControlNetPath { get; set; }
    }
}
