// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;

namespace TensorStack.Transformers.Processing
{
    public sealed class Sequence : IDisposable
    {
        private IKVCache _cache;

        public Sequence(IKVCache cache, long bos)
        {
            Tokens = [bos];
            _cache = cache;
        }

        private Sequence(List<long> tokens, float score, IKVCache cache)
        {
            Score = score;
            Tokens = tokens;
            _cache = cache;
        }

        public int Id { get; set; }
        public List<long> Tokens { get; }
        public float Score { get; set; }
        public bool IsComplete { get; set; }

        public int Length => Tokens.Count;
        public bool IsValid => !float.IsNegativeInfinity(Score);
        public OrtValue[] Cache => _cache.Values;


        public bool Initialize(int initialLength)
        {
            var isInitialized = _cache.IsInitialized;
            if (!isInitialized)
                _cache.Initialize(initialLength);
            return isInitialized;
        }


        public void UpdateCache(OrtValue[] currentValues, bool useBranchCache)
        {
            _cache.Update(currentValues, useBranchCache);
        }


        public Sequence Clone()
        {
            return new Sequence([.. Tokens], Score, _cache.Clone());
        }


        public void Dispose()
        {
            Tokens.Clear();
            _cache?.Dispose();
        }
    }
}
