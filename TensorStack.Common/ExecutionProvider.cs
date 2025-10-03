// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using Microsoft.ML.OnnxRuntime;

namespace TensorStack.Common
{
    public class ExecutionProvider
    {
        private readonly string _name;
        private readonly OrtMemoryInfo _memoryInfo;
     
        private readonly Func<ModelConfig, SessionOptions> _sessionOptionsFactory;

        public ExecutionProvider(string name, OrtMemoryInfo memoryInfo, Func<ModelConfig, SessionOptions> sessionOptionsFactory)
        {
            _name = name;
            _memoryInfo = memoryInfo;
            _sessionOptionsFactory = sessionOptionsFactory;
        }

        public string Name => _name;
        public OrtMemoryInfo MemoryInfo => _memoryInfo;

        public SessionOptions CreateSession(ModelConfig modelConfig)
        {
            return _sessionOptionsFactory(modelConfig);
        }
    }
}
