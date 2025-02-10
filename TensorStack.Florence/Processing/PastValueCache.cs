// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System;
using TensorStack.Common.Tensor;
using TensorStack.Core;

namespace TensorStack.Florence.Processing
{
    public sealed class PastValueCache : IDisposable
    {
        private readonly FlorenceConfig _configuration;
        private Tensor<float>[] _pastValues;

        /// <summary>
        /// Initializes a new instance of the <see cref="PastValueCache"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public PastValueCache(FlorenceConfig configuration)
        {
            _configuration = configuration;
        }

        /// <summary>
        /// Gets a value indicating whether this instance is initialized.
        /// </summary>
        public bool IsInitialized => _pastValues is not null;

        /// <summary>
        /// Gets the past values.
        /// </summary>
        /// <value>The past values.</value>
        public Tensor<float>[] PastValues => _pastValues;


        /// <summary>
        /// Initializes the cache with the specified batch size.
        /// </summary>
        /// <param name="batchSize">Size of the batch.</param>
        public void Initialize(int batchSize)
        {
            var encoderDimKv = _configuration.EncoderHiddenSize / _configuration.NumEncoderHeads;
            var decoderDimKv = _configuration.DecoderHiddenSize / _configuration.NumDecoderHeads;
            var encoderDims = new[] { batchSize, _configuration.NumEncoderHeads, 0, encoderDimKv };
            var decoderDims = new[] { batchSize, _configuration.NumDecoderHeads, 0, decoderDimKv };

            _pastValues = new Tensor<float>[_configuration.NumDecoderLayers * 4];
            for (var i = 0; i < _pastValues.Length; ++i)
            {
                if (i % 4 == 0)
                {
                    _pastValues[i] = new Tensor<float>(decoderDims);    // Decoder Key
                    _pastValues[i + 1] = new Tensor<float>(decoderDims);// Decoder Val
                    _pastValues[i + 2] = new Tensor<float>(encoderDims);// Encoder Key
                    _pastValues[i + 3] = new Tensor<float>(encoderDims);// Encoder Val
                }
            }
        }


        /// <summary>
        /// Updates the cache with the specified present values.
        /// </summary>
        /// <param name="presentValues">The present key values.</param>
        /// <param name="useCache">if set to <c>true</c> [use cache].</param>
        public void Update(OrtValue[] presentValues, bool useCache)
        {
            for (int i = 0; i < presentValues.Length; i++)
            {
                if (i % 4 == 0)
                {
                    _pastValues[i] = presentValues[i].ToTensor();        // Decoder Key
                    _pastValues[i + 1] = presentValues[i + 1].ToTensor();// Decoder Val
                    if (!useCache)
                    {
                        _pastValues[i + 2] = presentValues[i + 2].ToTensor();// Encoder Key
                        _pastValues[i + 3] = presentValues[i + 3].ToTensor();// Encoder Val
                    }
                }
            }
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            _pastValues = null;
        }

    }
}
