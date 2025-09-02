// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System;

namespace TensorStack.Transformers.Processing
{
    public interface IKVCache : IDisposable
    {
        bool IsInitialized { get; }
        OrtValue[] Values { get; }
        void Update(OrtValue[] currentValues, bool useBranchCache);
        void Initialize(int initialSize);
        IKVCache Clone();
    }
}
