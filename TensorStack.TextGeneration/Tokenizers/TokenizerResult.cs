// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common.Tensor;

namespace TensorStack.TextGeneration.Tokenizers
{
    public record TokenizerResult
    {
        public TokenizerResult(Tensor<long> inputIds, Tensor<long> mask, Tensor<float> weights = default, string normalizedInput = default)
        {
            Mask = mask;
            Weights = weights;
            InputIds = inputIds;
            NormalizedInput = normalizedInput;
        }

        public TokenizerResult(long[] inputIds, long[] mask, float[] weights = default, string normalizedInput = default)
        {
            Mask = new Tensor<long>(mask, [1, mask.Length]);
            InputIds = new Tensor<long>(inputIds, [1, inputIds.Length]);
            NormalizedInput = normalizedInput;
            if (weights != null)
                Weights = new Tensor<float>(weights, [1, weights.Length]);
        }

        public Tensor<long> Mask { get; set; }
        public Tensor<long> InputIds { get; set; }
        public Tensor<float> Weights { get; set; }
        public string NormalizedInput { get; set; }
        public int Length => (int)InputIds.Length;
    }
}
