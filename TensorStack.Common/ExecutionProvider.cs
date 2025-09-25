// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using Microsoft.ML.OnnxRuntime;

namespace TensorStack.Common
{
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
