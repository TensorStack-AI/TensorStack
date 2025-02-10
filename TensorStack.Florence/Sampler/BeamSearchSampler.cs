// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.Florence.Common;

namespace TensorStack.Florence.Sampler
{
    //TODO: Better Beam search, Optimize TopK
    public class BeamSearchSampler
    {
        private readonly int _topK;
        private readonly Random _random;
        private record TopkResult(Tensor<long> I, Tensor<float> V);

        /// <summary>
        /// Initializes a new instance of the <see cref="BeamSearchSampler"/> class.
        /// </summary>
        /// <param name="topK">The top k.</param>
        /// <param name="seed">The seed.</param>
        public BeamSearchSampler(int topK, int seed = 0)
        {
            _topK = topK;
            _random = new Random(seed);
        }


        /// <summary>
        /// Samples the specified logits.
        /// </summary>
        /// <param name="inputIds">The input ids.</param>
        /// <param name="logits">The logits.</param>
        /// <returns></returns>
        public IEnumerable<LogitsResult> Sample(List<long>[] inputIds, Tensor<float> logits)
        {
            var k = logits.Dimensions[^1];
            if (_topK > 0)
                k = Math.Min(_topK, k);

            // Compute TopK
            var topkResult = GetTopK(logits, k);

            // Compute softmax
            var probabilities = topkResult.V.SoftMax();

            // Sample Beams
            var numBeams = inputIds.Length;
            for (int x = 0; x < numBeams; x++)
            {
                var index = GetRandomSample(x, inputIds, probabilities);
                yield return new LogitsResult
                (
                    topkResult.I[x, index], // TokenId
                    MathF.Log(probabilities[x, index]) // Score
                );
            }
        }


        /// <summary>
        /// Gets the TopK indacies and values.
        /// </summary>
        /// <param name="logits">The logits.</param>
        /// <param name="k">The k.</param>
        /// <returns></returns>
        private TopkResult GetTopK(Tensor<float> logits, int k)
        {
            var numBeams = logits.Dimensions[0];
            var vocabSize = logits.Dimensions[^1];
            var indices = new Tensor<long>([numBeams, k]);
            var topKLogits = new Tensor<float>([numBeams, k]);
            for (int beam = 0; beam < numBeams; beam++)
            {
                var beamLogits = logits.GetBatchAsSpan(beam).ToArray();
                var topKResult = beamLogits
                    .Select((v, i) => new { V = v, I = i })
                    .OrderByDescending(x => x.V)
                    .Take(k)
                    .ToArray();

                for (int i = 0; i < k; i++)
                {
                    indices[beam, i] = topKResult[i].I;
                    topKLogits[beam, i] = topKResult[i].V;
                }
            }
            return new TopkResult(indices, topKLogits);
        }


        /// <summary>
        /// Gets the next random sample.
        /// </summary>
        /// <param name="beamIndex">Index of the beam.</param>
        /// <param name="inputIds">The input ids.</param>
        /// <param name="probabilities">The probabilities.</param>
        /// <returns></returns>
        private int GetRandomSample(int beamIndex, List<long>[] inputIds, Tensor<float> probabilities)
        {
            var iteration = inputIds[beamIndex].Count;
            if (beamIndex == 0 || iteration == 0)
                return 0; // 1st beam greedy

            var cumulative = 0f;
            var randomValue = _random.NextSingle();
            var beamProbabilities = probabilities.GetBatchAsSpan(beamIndex);
            for (int i = 0; i < beamProbabilities.Length; i++)
            {
                cumulative += beamProbabilities[i];
                if (randomValue < cumulative)
                    return i;
            }
            return 0;
        }

    }
}