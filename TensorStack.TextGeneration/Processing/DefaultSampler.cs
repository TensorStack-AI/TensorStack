// Copyright(c) TensorStack.All rights reserved. // Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using TensorStack.Common;
using TensorStack.Common.Tensor;
namespace TensorStack.TextGeneration.Processing
{
    public class DefaultSampler
    {
        private readonly Random _random;

        /// <summary>
        /// Initializes a new instance of the <see cref="DefaultSampler"/> class.
        /// </summary>
        /// <param name="seed">The seed.</param>
        public DefaultSampler(int seed = -1)
        {
            _random = seed < 0
                ? new Random()
                : new Random(seed);
        }


        /// <summary>
        /// Samples the specified logits.
        /// </summary>
        /// <param name="logits">The logits.</param>
        /// <param name="topK">The top k.</param>
        /// <param name="topP">The top p.</param>
        /// <param name="temperature">The temperature.</param>
        /// <returns>LogitResult[].</returns>
        public LogitResult[] Sample(Tensor<float> logits, int topK = 1, float topP = 1f, float temperature = 1f)
        {
            ApplyTemperature(logits, temperature);

            var topkLogits = SelectTopK(logits, topK);

            var probabilities = GetProbabilities(topkLogits);

            var candidates = SelectTopP(probabilities, topP);

            var results = new LogitResult[candidates.Length];
            for (int i = 0; i < candidates.Length; i++)
                results[i] = SampleNext(candidates);

            return results;
        }


        /// <summary>
        /// Gets the probabilities.
        /// </summary>
        /// <param name="topkLogits">The topk logits.</param>
        /// <returns>Span&lt;LogitResult&gt;.</returns>
        private Span<LogitResult> GetProbabilities(TopkResult topkLogits)
        {
            var probabilities = topkLogits.V.SoftMax();
            var logitProbabilities = new LogitResult[probabilities.Dimensions[1]].AsSpan();
            for (int i = 0; i < logitProbabilities.Length; i++)
                logitProbabilities[i] = new LogitResult(topkLogits.I[0, i], probabilities[0, i]);

            return logitProbabilities;
        }


        /// <summary>
        /// Selects the TopK candidates.
        /// </summary>
        /// <param name="logits">The logits.</param>
        /// <param name="topk">The topk.</param>
        /// <returns>TopkResult.</returns>
        private TopkResult SelectTopK(Tensor<float> logits, int topk)
        {
            var indices = new Tensor<long>([1, topk]);
            var topKLogits = new Tensor<float>([1, topk]);
            var queue = new PriorityQueue<(float v, int i), float>();
            for (int i = 0; i < logits.Length; i++)
            {
                var logit = logits[0, i];
                queue.Enqueue((logit, i), logit);
                if (queue.Count > topk)
                    queue.Dequeue();
            }

            for (int i = topk - 1; i >= 0; i--)
            {
                var (v, idx) = queue.Dequeue();
                indices[0, i] = idx;
                topKLogits[0, i] = v;
            }
            return new TopkResult(indices, topKLogits);
        }


        /// <summary>
        /// Selects the TopP candidates.
        /// </summary>
        /// <param name="candidates">The candidates.</param>
        /// <param name="topP">The top p.</param>
        /// <returns>Span&lt;LogitResult&gt;.</returns>
        private Span<LogitResult> SelectTopP(Span<LogitResult> candidates, float topP)
        {
            if (topP < 1f)
            {
                var cumulative = 0f;
                var filtered = new List<LogitResult>();
                foreach (var candidate in candidates)
                {
                    cumulative += candidate.Score;
                    filtered.Add(candidate);
                    if (cumulative >= topP)
                        break;
                }

                foreach (var result in filtered)
                    result.Score /= cumulative;

                return filtered.ToArray();
            }
            return candidates;
        }


        /// <summary>
        /// Samples the next token.
        /// </summary>
        /// <param name="candidates">The candidates.</param>
        /// <returns>LogitResult.</returns>
        private LogitResult SampleNext(Span<LogitResult> candidates)
        {
            if (candidates.Length == 1)
                return candidates[0];

            var cumulative = 0f;
            var random = _random.NextSingle();
            foreach (var c in candidates)
            {
                cumulative += c.Score;
                if (random < cumulative)
                    return c;
            }
            return candidates[^1]; // fallback
        }


        /// <summary>
        /// Applies the temperature.
        /// </summary>
        /// <param name="logits">The logits.</param>
        /// <param name="temperature">The temperature.</param>
        private void ApplyTemperature(Tensor<float> logits, float temperature)
        {
            if (temperature != 1.0f)
            {
                var span = logits.Memory.Span;
                for (int i = 0; i < span.Length; i++)
                    span[i] /= temperature;
            }
        }


        private record TopkResult(Tensor<long> I, Tensor<float> V);
    }
    public record LogitResult
    {
        public LogitResult(long tokenId, float score)
        {
            Score = score;
            TokenId = tokenId;
        }
        public long TokenId { get; }
        public float Score { get; set; }
    };
}