// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.Transformers.Common;

namespace TensorStack.Transformers.Processing
{
    public class DefaultSampler
    {
        public IEnumerable<LogitsResult> Sample(Tensor<float> logits, int topK = 1, float topP = 1f, float temperature = 1f)
        {
            ApplyTemperature(logits, temperature);

            var topkResult = SelectTopK(logits, topK);
            var probabilities = topkResult.V.SoftMax();

            if (topP >= 1f)
            {
                for (int i = 0; i < topK; i++)
                {
                    yield return new LogitsResult
                    (
                        topkResult.I[0, i],
                        MathF.Log(probabilities[0, i])
                    );
                }
            }
            else
            {
                var sorted = new List<LogitsResult>();
                for (int i = 0; i < probabilities.Dimensions[1]; i++)
                    sorted.Add(new LogitsResult(topkResult.I[0, i], probabilities[0, i]));

                sorted.Sort((a, b) => b.Score.CompareTo(a.Score));

                var cumulative = 0f;
                var filtered = new List<LogitsResult>();
                foreach (var item in sorted)
                {
                    cumulative += item.Score;
                    filtered.Add(item);
                    if (cumulative >= topP)
                        break;
                }

                foreach (var result in filtered)
                {
                    yield return new LogitsResult(result.TokenId, MathF.Log(result.Score / cumulative));
                }
            }
        }


        private TopkResult SelectTopK(Tensor<float> logits, int topk)
        {
            var indices = new Tensor<long>([1, topk]);
            var topKLogits = new Tensor<float>([1, topk]);
            var batchLogits = logits.GetBatchAsSpan(0);
            var queue = new PriorityQueue<(float v, int i), float>();

            for (int i = 0; i < batchLogits.Length; i++)
            {
                queue.Enqueue((batchLogits[i], i), batchLogits[i]);
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

    public record LogitsResult(long TokenId, float Score);
}