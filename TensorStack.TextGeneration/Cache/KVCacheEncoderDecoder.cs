// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using TensorStack.Common;
using Metadata = TensorStack.Common.ModelMetadata;

namespace TensorStack.TextGeneration.Cache
{
    public sealed class KVCacheEncoderDecoder : IKVCache
    {
        private readonly Metadata _metadata;
        private readonly int _numHeads;
        private readonly int _numLayers;
        private readonly int _hiddenSize;
        private readonly int _headDimension;
        private OrtValue[] _values;


        /// <summary>
        /// Initializes a new instance of the <see cref="KVCacheEncoderDecoder"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public KVCacheEncoderDecoder(Metadata metadata, int numHeads, int numLayers, int hiddenSize)
        {
            _metadata = metadata;
            _numHeads = numHeads;
            _numLayers = numLayers;
            _hiddenSize = hiddenSize;
            _headDimension = _hiddenSize / _numHeads;
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="KVCacheEncoderDecoder"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="cacheValues">The past values.</param>
        private KVCacheEncoderDecoder(Metadata metadata, int numHeads, int numLayers, int hiddenSize, OrtValue[] values)
            : this(metadata, numHeads, numLayers, hiddenSize)
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
            _values = new OrtValue[_numLayers * 4];
            var allocator = _metadata.Allocator;
            var elementType = _metadata.OutputElementType;
            var decoderDims = new[] { 1L, _numHeads, initialSize, _headDimension };
            var encoderDims = new[] { 1L, _numHeads, initialSize, _headDimension };
            for (var i = 0; i < _values.Length; ++i)
            {
                if (i % 4 == 0)
                {
                    _values[i] = OrtValue.CreateAllocatedTensorValue(allocator, elementType, decoderDims);    // Decoder Key
                    _values[i + 1] = OrtValue.CreateAllocatedTensorValue(allocator, elementType, decoderDims);// Decoder Val
                    _values[i + 2] = OrtValue.CreateAllocatedTensorValue(allocator, elementType, encoderDims);// Encoder Key
                    _values[i + 3] = OrtValue.CreateAllocatedTensorValue(allocator, elementType, encoderDims);// Encoder Val
                }
            }
        }


        /// <summary>
        /// Updates the cache with the specified present values.
        /// </summary>
        /// <param name="currentValues">The current key values.</param>
        /// <param name="useBranchCache">if set to <c>true</c> [use cache].</param>
        public void Update(OrtValue[] currentValues, bool useBranchCache)
        {
            for (int i = 0; i < currentValues.Length; i++)
            {
                if (i % 4 == 0)
                {
                    // TODO: Allocate entire Maxlength and update the buffer

                    // Decoder Key
                    _values[i].Dispose();
                    _values[i] = currentValues[i];

                    // Decoder Val
                    _values[i + 1].Dispose();
                    _values[i + 1] = currentValues[i + 1];

                    if (!useBranchCache)
                    {
                        _values[i + 2] = currentValues[i + 2];// Encoder Key
                        _values[i + 3] = currentValues[i + 3];// Encoder Val
                    }
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
                cacheValues[i] = _values[i].Clone(_metadata.Allocator);  // TODO: Slice

            return new KVCacheEncoderDecoder(_metadata, _numHeads, _numLayers, _hiddenSize, cacheValues);
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
