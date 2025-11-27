// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Collections.Generic;
using System.Text.Json.Serialization;

namespace TensorStack.Common
{
    public record ModelConfig
    {
        private ExecutionProvider _executionProvider;

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public string Path { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public bool IsOptimizationSupported { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingDefault)]
        public Dictionary<string, string> SessionOptions { get; set; }

        [JsonIgnore(Condition = JsonIgnoreCondition.WhenWritingNull)]
        public ExecutionProvider ExecutionProvider
        {
            get { return _executionProvider; }
            init { _executionProvider = value; }
        }

        public virtual void SetProvider(ExecutionProvider executionProvider)
        {
            _executionProvider = executionProvider;
        }
    }
}
