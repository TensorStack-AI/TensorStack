// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using TensorStack.Common;
using OrtType = Microsoft.ML.OnnxRuntime.Tensors.TensorElementType;

namespace TensorStack.TextGeneration.Processing
{
    public sealed class KVCacheDecoder : IKVCache
    {
        private readonly OrtType _dataType;
        private readonly int _numHeads;
        private readonly int _numLayers;
        private readonly int _hiddenSize;
        private readonly int _numKVHeads;
        private OrtValue[] _values;


        /// <summary>
        /// Initializes a new instance of the <see cref="KVCacheDecoder"/> class.
        /// </summary>
        /// <param name="dataType">Type of the data.</param>
        /// <param name="numHeads">The number heads.</param>
        /// <param name="numLayers">The number layers.</param>
        /// <param name="hiddenSize">Size of the hidden.</param>
        public KVCacheDecoder(OrtType dataType, int numHeads, int numLayers, int hiddenSize, int numKVHeads)
        {
            _dataType = dataType;
            _numHeads = numHeads;
            _numLayers = numLayers;
            _hiddenSize = hiddenSize;
            _numKVHeads = numKVHeads;
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="KVCacheDecoder"/> class.
        /// </summary>
        /// <param name="dataType">Type of the data.</param>
        /// <param name="numHeads">The number heads.</param>
        /// <param name="numLayers">The number layers.</param>
        /// <param name="hiddenSize">Size of the hidden.</param>
        /// <param name="values">The cache values.</param>
        private KVCacheDecoder(OrtType dataType, int numHeads, int numLayers, int hiddenSize, int numKVHeads, OrtValue[] values)
            : this(dataType, numHeads, numLayers, hiddenSize, numKVHeads)
        {
            _values = values;
        }


        /// <summary>
        /// Gets a value indicating whether this instance is initialized.
        /// </summary>
        public bool IsInitialized => _values is not null;

        /// <summary>
        /// Gets the cache values.
        /// </summary>
        public OrtValue[] Values => _values;


        /// <summary>
        /// Initializes the cache with the specified batch size.
        /// </summary>
        /// <param name="batchSize">Size of the batch.</param>
        public void Initialize(int initialSize)
        {
            _values = new OrtValue[_numLayers * 2];
            var dimensions = new[] { 1L, _numKVHeads, 1, (_hiddenSize / _numHeads) };
            for (var i = 0; i < _values.Length; ++i)
            {
                if (i % 2 == 0)
                {
                    _values[i] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, _dataType, dimensions);
                    _values[i + 1] = OrtValue.CreateAllocatedTensorValue(OrtAllocator.DefaultInstance, _dataType, dimensions);
                }
            }
        }


        /// <summary>
        /// Updates the cache with the specified present values.
        /// </summary>
        /// <param name="currentValues">The current key values.</param>
        /// <param name="useCache">if set to <c>true</c> [use cache].</param>
        public void Update(OrtValue[] currentValues, bool useBranchCache)
        {
            for (int i = 0; i < currentValues.Length; i++)
            {
                if (i % 2 == 0)
                {
                    _values[i].Dispose();
                    _values[i + 1].Dispose();

                    _values[i] = currentValues[i];        // Decoder Key
                    _values[i + 1] = currentValues[i + 1];// Decoder Val
                }
            }
        }


        /// <summary>
        /// Clones this instance.
        /// </summary>
        /// <returns>EncoderDecoderKVCache.</returns>
        public IKVCache Clone()
        {
            var cacheValues = new OrtValue[_values.Length];
            for (int i = 0; i < _values.Length; i++)
                cacheValues[i] = _values[i].Clone();

            return new KVCacheDecoder(_dataType, _numHeads, _numLayers, _hiddenSize, _numKVHeads, cacheValues);
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            foreach (var cacheValue in _values)
                cacheValue?.Dispose();

            _values = null;
        }
    }
}
