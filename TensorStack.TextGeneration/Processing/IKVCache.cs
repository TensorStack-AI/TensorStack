// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System;

namespace TensorStack.TextGeneration.Processing
{
    public interface IKVCache : IDisposable
    {
        /// <summary>
        /// Gets a value indicating whether this sequence is initialized.
        /// </summary>
        bool IsInitialized { get; }

        /// <summary>
        /// Gets the cache values.
        /// </summary>
        OrtValue[] Values { get; }

        /// <summary>
        /// Updates the current cache values.
        /// </summary>
        /// <param name="currentValues">The current values.</param>
        /// <param name="useBranchCache">if set to <c>true</c> use branch cache.</param>
        void Update(OrtValue[] currentValues, bool useBranchCache);

        /// <summary>
        /// Initializes the cache with the specified initial size.
        /// </summary>
        /// <param name="initialSize">The initial size.</param>
        void Initialize(int initialSize);

        /// <summary>
        /// Clones this cache instance.
        /// </summary>
        /// <returns>IKVCache.</returns>
        IKVCache Clone();
    }
}
