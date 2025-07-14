// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Linq;

namespace TensorStack.StableDiffusion.Common
{
    public record TokenizerResult
    {
        public TokenizerResult(long[] inputIds, long[] attentionMask)
        {
            InputIds = inputIds;
            AttentionMask = attentionMask;
            Weights = [.. Enumerable.Repeat(1f, inputIds.Length)];
        }

        public TokenizerResult(long[] inputIds, long[] attentionMask, float[] weights)
        {
            InputIds = inputIds;
            AttentionMask = attentionMask;
            Weights = weights;
        }

        public long[] InputIds { get; set; }
        public long[] AttentionMask { get; set; }
        public float[] Weights { get; set; }
    }
}
