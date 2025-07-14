// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Text.Json.Serialization;
using TensorStack.Common;
using TensorStack.StableDiffusion.Enums;

namespace TensorStack.StableDiffusion.Config
{
    public record UNetModelConfig : ModelConfig
    {
        public int InChannels { get; set; } = 4;
        public int OutChannels { get; set; } = 4;

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public ModelType ModelType { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public string ControlNetPath { get; set; }
    }
}
