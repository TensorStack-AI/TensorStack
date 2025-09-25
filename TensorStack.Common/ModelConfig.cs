// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Text.Json.Serialization;

namespace TensorStack.Common
{
    public record ModelConfig
    {
        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public string Path { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public bool IsOptimizationSupported { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public ExecutionProvider ExecutionProvider { get; private set; }

        public virtual void SetProvider(ExecutionProvider executionProvider)
        {
            ExecutionProvider = executionProvider;
        }
    }
}
