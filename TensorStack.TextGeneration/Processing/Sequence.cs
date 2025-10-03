// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;

namespace TensorStack.TextGeneration.Processing
{
    public sealed class Sequence : IDisposable
    {
        private IKVCache _cache;

        /// <summary>
        /// Initializes a new instance of the <see cref="Sequence"/> class.
        /// </summary>
        /// <param name="cache">The cache.</param>
        /// <param name="bos">The bos.</param>
        public Sequence(IKVCache cache, params List<long> startSequence)
        {
            _cache = cache;
            Tokens = startSequence;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="Sequence"/> class.
        /// </summary>
        /// <param name="tokens">The tokens.</param>
        /// <param name="score">The score.</param>
        /// <param name="cache">The cache.</param>
        private Sequence(List<long> tokens, float score, IKVCache cache)
        {
            Score = score;
            Tokens = tokens;
            _cache = cache;
        }

        /// <summary>
        /// Gets the tokens.
        /// </summary>
        public List<long> Tokens { get; }

        /// <summary>
        /// Gets or sets a value indicating whether this sequence is complete.
        /// </summary>
        public bool IsComplete { get; set; }

        /// <summary>
        /// Gets the sequence length.
        /// </summary>
        public int Length => Tokens.Count;

        /// <summary>
        /// Gets the cache.
        /// </summary>
        public OrtValue[] Cache => _cache.Values;

        /// <summary>
        /// Gets or sets the sequnece score.
        /// </summary>
        public float Score { get; set; }

        /// <summary>
        /// Gets or sets the penalty score.
        /// </summary>
        public float PenaltyScore { get; set; }

        /// <summary>
        /// Returns true if the sequence is valid.
        /// </summary>
        public bool IsValid => !float.IsNegativeInfinity(Score);


        /// <summary>
        /// Initializes the sequence with the specified initial length.
        /// </summary>
        /// <param name="initialLength">The initial length.</param>
        public bool Initialize(int initialLength)
        {
            var isInitialized = _cache.IsInitialized;
            if (!isInitialized)
                _cache.Initialize(initialLength);
            return isInitialized;
        }


        /// <summary>
        /// Updates the cache.
        /// </summary>
        /// <param name="currentValues">The current values.</param>
        /// <param name="useBranchCache">if set to <c>true</c> use branch cache.</param>
        public void UpdateCache(OrtValue[] currentValues, bool useBranchCache)
        {
            _cache.Update(currentValues, useBranchCache);
        }


        /// <summary>
        /// Clones this sequence.
        /// </summary>
        /// <returns>Sequence.</returns>
        public Sequence Clone()
        {
            return new Sequence([.. Tokens], Score, _cache.Clone());
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            Tokens.Clear();
            _cache?.Dispose();
            _cache = null;
        }
    }
}
