// Copyright(c) TensorStack.All rights reserved. // Licensed under the Apache 2.0 License.
using System;
using TensorStack.Common.Tensor;
using TensorStack.TextGeneration;
using TensorStack.TextGeneration.Common;

namespace TensorStack.TextGeneration.Processing.Sampler
{
    public class MultinomialSampler : Sampler
    {
        private readonly Random _random;

        /// <summary>
        /// Initializes a new instance of the <see cref="MultinomialSampler "/> class.
        /// </summary>
        /// <param name="seed">The seed.</param>
        public MultinomialSampler(GenerateOptions options) 
            : base(options) 
        {
            _random = options.Seed < 0
                ? new Random()
                : new Random(options.Seed);
        }


        /// <summary>
        /// Samples the specified logits.
        /// </summary>
        /// <param name="logits">The logits.</param>
        /// <param name="topK">The top k.</param>
        /// <param name="topP">The top p.</param>
        /// <param name="temperature">The temperature.</param>
        /// <returns>LogitResult[].</returns>
        public override LogitResult[] Sample(Tensor<float> logits, int topK = 1, float topP = 1f, float temperature = 1f)
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

    }
}