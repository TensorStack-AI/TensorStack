// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.Transformers.Common;
using TensorStack.Transformers.Processing;
using TensorStack.Transformers.Tokenizers;

namespace TensorStack.Transformers.Pipelines
{
    public abstract class PipelineBase : IDisposable
    {
        private readonly DecoderConfig _decoderConfig;
        private readonly T5TokenizerConfig _tokenizerConfig;

        private readonly ModelSession _decoder;
        private readonly T5Tokenizer _tokenizer;
        private readonly DefaultSampler _sampler;

        /// <summary>
        /// Initializes a new instance of the <see cref="Phi3Pipeline"/> class.
        /// </summary>
        /// <param name="tokenizerConfig">The tokenizer configuration.</param>
        /// <param name="decoderConfig">The decoder configuration.</param>
        public PipelineBase(T5TokenizerConfig tokenizerConfig, DecoderConfig decoderConfig)
        {
            _decoderConfig = decoderConfig;
            _tokenizerConfig = tokenizerConfig;

            _sampler = new DefaultSampler();
            _tokenizer = new T5Tokenizer(_tokenizerConfig);
            _decoder = new ModelSession(_decoderConfig);
        }

        protected T5Tokenizer Tokenizer => _tokenizer;
        protected T5TokenizerConfig TokenizerConfig => _tokenizerConfig;
        protected ModelSession Decoder => _decoder;
        protected DecoderConfig DecoderConfig => _decoderConfig;
        protected DefaultSampler Sampler => _sampler;
        protected T5TokenizerResult TokenizerOutput { get; set; }

        protected abstract Task<Sequence> InitializeAsync(GenerateOptions options);
        protected abstract Task<Tensor<float>> RunDecoderAsync(Sequence sequence);


        /// <summary>
        /// Loads the models.
        /// </summary>
        public virtual async Task LoadAsync(CancellationToken cancellationToken = default)
        {
            await _decoder.LoadAsync(cancellationToken: cancellationToken);
        }


        /// <summary>
        /// Unloads the models.
        /// </summary>
        public virtual async Task UnloadAsync(CancellationToken cancellationToken = default)
        {
            await _decoder.UnloadAsync();
        }


        protected virtual ILogitsProcessor[] GetLogitsProcessor(GenerateOptions options)
        {
            return
            [
               new BOSLogitsProcessor(_tokenizer.BOS),
               new NoRepeatNGramLogitsProcessor(options.NoRepeatNgramSize)
            ];
        }


        protected virtual ITokenProcessor[] GetTokenProcessors(GenerateOptions options)
        {
            return
            [
                new EOSTokenProcessor(options.MinLength, _tokenizer.EOS),
                new MaxLengthTokenProcessor(options.MaxLength)
            ];
        }


        protected virtual async Task<Sequence[]> BeamSearchAsync(GenerateOptions options, CancellationToken cancellationToken = default)
        {
            var logitsProcessors = GetLogitsProcessor(options);
            var tokenProcessors = GetTokenProcessors(options);

            var initialSequence = await InitializeAsync(options);
            var activeBeams = new List<Sequence>(options.Beams) { initialSequence };
            while (!cancellationToken.IsCancellationRequested)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var beamId = 0;
                var beamCandidates = new List<Sequence>();
                foreach (var beam in activeBeams)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    if (beam.IsComplete)
                    {
                        beamCandidates.Add(beam);
                        continue;
                    }

                    // Compute Logits
                    var logits = await RunDecoderAsync(beam);

                    // Logit Processors
                    foreach (var logitsProcessor in logitsProcessors)
                        logitsProcessor.Process(beam.Tokens, logits);

                    // Sample
                    var samples = Sampler.Sample(logits, options.TopK, options.TopP, options.Temperature);
                    foreach (var sample in samples)
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        var beamCandidate = options.TopK > 1 ? beam.Clone() : beam;
                        beamCandidate.Id = beamId;
                        beamCandidate.Tokens.Add(sample.TokenId);
                        beamCandidate.Score += sample.Score;
                        beamCandidates.Add(beamCandidate);
                    }

                    beamId++;
                }

                // Select Beams
                activeBeams.Clear();
                activeBeams.AddRange(GetSequenceCandidates(beamCandidates, options));

                // Process Beams
                foreach (var beam in activeBeams)
                {
                    Console.WriteLine(Tokenizer.Decode(beam.Tokens));

                    if (beam.IsComplete)
                        continue;

                    if (tokenProcessors.Any(x => x.Process(beam)))
                    {
                       // Console.Write(" [**Beam Complete**]");
                        beam.IsComplete = true;
                    }
                }

                Console.WriteLine("-------------");

                // Check Completion
                if (activeBeams.All(x => x.IsComplete))
                    break;
            }

            // Return beam reuslts
            return NormalizeAndSort(activeBeams, options);
        }


        protected virtual IEnumerable<Sequence> GetSequenceCandidates(List<Sequence> sequences, GenerateOptions options)
        {
            // TODO: Diversity Penalty

            return sequences
                .OrderByDescending(s => GetLengthPenalty(s, options.LengthPenalty))
                .Take(options.Beams);
        }


        protected virtual float GetLengthPenalty(Sequence sequence, float penalty)
        {
            return sequence.Score / MathF.Pow((5.0f + sequence.Length) / 6.0f, penalty);
        }


        protected virtual Sequence[] NormalizeAndSort(List<Sequence> sequences, GenerateOptions options)
        {
            // Normalize
            var beam = 0;
            var minScore = sequences.Min(b => b.Score);
            var maxScore = sequences.Max(b => b.Score);
            var scoreRange = maxScore - minScore;
            foreach (var sequence in sequences)
            {
                sequence.Id = beam++;
                sequence.Score = scoreRange > 0 ? (sequence.Score - minScore) / scoreRange : 1f;
            }

            // Sort
            sequences.Sort((a, b) => b.Score.CompareTo(a.Score));
            return [.. sequences];
        }


        protected virtual Tensor<long> GetPositionIds(ModelMetadata metadata, int startPosition, int endPosition = 0)
        {
            var hasPositionIds = metadata.Inputs.Count > ((_decoderConfig.NumLayers * 2) + 2);
            if (!hasPositionIds)
                return default;

            if (endPosition == 0)
                return new Tensor<long>(new long[] { startPosition }, [1, 1]);

            var positionIds = Enumerable.Range(startPosition, endPosition)
                .Select(Convert.ToInt64)
                .ToArray();
            return new Tensor<long>(positionIds, [1, positionIds.Length]);
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            Dispose(true);
            GC.SuppressFinalize(this);
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        private bool _disposed;
        protected virtual void Dispose(bool disposing)
        {
            if (_disposed)
                return;

            if (disposing)
            {
                _tokenizer.Dispose();
                _decoder.Dispose();
            }

            _disposed = true;
        }
    }


    public static class SequenceExtensions
    {
        // Returns how many tokens are the same at the start of both sequences
        public static int CommonPrefixLength(this IReadOnlyList<long> a, IReadOnlyList<long> b)
        {
            int minLength = Math.Min(a.Count, b.Count);
            int count = 0;
            for (int i = 0; i < minLength; i++)
            {
                if (a[i] != b[i]) break;
                count++;
            }
            return count;
        }
    }
}