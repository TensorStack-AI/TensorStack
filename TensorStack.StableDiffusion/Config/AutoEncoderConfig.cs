// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Text.Json.Serialization;
using TensorStack.Common;

namespace TensorStack.StableDiffusion.Config
{
    public record AutoEncoderModelConfig : ModelConfig
    {
        public int Scale { get; set; } = 8;
        public float ScaleFactor { get; set; }
        public float ShiftFactor { get; set; }
        public int InChannels { get; set; } = 3;
        public int OutChannels { get; set; } = 3;
        public int LatentChannels { get; set; } = 4;
        public string DecoderModelPath { get; set; }
        public string EncoderModelPath { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public float[] LatentsStd { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public float[] LatentsMean {get; set; }
}
}
