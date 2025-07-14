// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Text.Json.Serialization;
using Microsoft.ML.OnnxRuntime;

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

    public class ExecutionProvider
    {
        private readonly string _name;
        private readonly Func<ModelConfig, SessionOptions> _sessionOptionsFactory;

        public ExecutionProvider(string name, Func<ModelConfig, SessionOptions> sessionOptionsFactory)
        {
            _name = name;
            _sessionOptionsFactory = sessionOptionsFactory;
        }

        public string Name => _name;

        public SessionOptions CreateSession(ModelConfig modelConfig)
        {
            return _sessionOptionsFactory(modelConfig);
        }
    }
}
