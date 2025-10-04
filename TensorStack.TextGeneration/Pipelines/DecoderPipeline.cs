// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.TextGeneration.Common;
using TensorStack.TextGeneration.Pipelines.Phi;
using TensorStack.TextGeneration.Processing;
using TensorStack.TextGeneration.Sampler;
using TensorStack.TextGeneration.Tokenizers;

namespace TensorStack.TextGeneration.Pipelines
{
    public abstract class DecoderPipeline<O> : IDisposable where O : GenerateOptions
    {
        private readonly DecoderConfig _decoderConfig;
        private readonly ModelSession _decoder;
        private readonly ITokenizer _tokenizer;
        private readonly SequenceComparer _sequenceComparer;

        /// <summary>
        /// Initializes a new instance of the <see cref="Phi3Pipeline"/> class.
        /// </summary>
        /// <param name="tokenizerConfig">The tokenizer configuration.</param>
        /// <param name="decoderConfig">The decoder configuration.</param>
        public DecoderPipeline(ITokenizer tokenizer, DecoderConfig decoderConfig)
        {
            _tokenizer = tokenizer;
            _decoderConfig = decoderConfig;
            _decoder = new ModelSession(_decoderConfig);
            _sequenceComparer = new SequenceComparer(_tokenizer.SpecialTokens, 5);
        }

        protected ITokenizer Tokenizer => _tokenizer;
        protected ModelSession Decoder => _decoder;
        protected DecoderConfig DecoderConfig => _decoderConfig;
        protected TokenizerResult TokenizerOutput { get; set; }
        protected abstract Task<Sequence> InitializeAsync(O options);
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


        /// <summary>
        /// Gets the logits processors.
        /// </summary>
        /// <param name="options">The options.</param>
        protected virtual ILogitsProcessor[] GetLogitsProcessor(O options)
        {
            return
            [
               new BOSLogitsProcessor(_tokenizer.BOS),
               new NoRepeatNGramLogitsProcessor(options.NoRepeatNgramSize)
            ];
        }


        /// <summary>
        /// Gets the token processors.
        /// </summary>
        /// <param name="options">The options.</param>
        protected virtual ITokenProcessor[] GetTokenProcessors(O options)
        {
            return
            [
                new EOSTokenProcessor(options.MinLength, _tokenizer.EOS),
                new MaxLengthTokenProcessor(options.MaxLength)
            ];
        }


        /// <summary>
        /// Tokenize the prompt
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns>A Task representing the asynchronous operation.</returns>
        protected virtual async Task TokenizePromptAsync(O options)
        {
            TokenizerOutput = await Tokenizer.EncodeAsync(options.Prompt);
        }


        /// <summary>
        /// Gets the sampler.
        /// </summary>
        /// <param name="options">The options.</param>
        protected virtual SamplerBase GetSampler(O options, bool isBeamSerach)
        {
            return isBeamSerach
                ? new MultinomialSampler(options)
                : new GreedySampler(options);
        }


        /// <summary>
        /// Greedy search
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;Sequence&gt; representing the asynchronous operation.</returns>
        protected virtual async Task<Sequence> GreedySearchAsync(O options, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var sampler = GetSampler(options, false);
            var logitsProcessors = GetLogitsProcessor(options);
            var tokenProcessors = GetTokenProcessors(options);

            var sequence = await InitializeAsync(options);
            while (!cancellationToken.IsCancellationRequested)
            {
                cancellationToken.ThrowIfCancellationRequested();

                // Compute Logits
                var logits = await RunDecoderAsync(sequence);

                // Logit Processors
                foreach (var logitsProcessor in logitsProcessors)
                    logitsProcessor.Process(sequence.Tokens, logits);

                // Sample
                var sample = sampler.Sample(logits, temperature: options.Temperature).First();
                sequence.Tokens.Add(sample.TokenId);
                sequence.Score += sample.Score;

                // Notify
                NotifyProgress(progressCallback, sequence);

                // Check Completion
                if (tokenProcessors.Any(x => x.Process(sequence)))
                    break;
            }

            // Return reuslt
            return sequence;
        }


        /// <summary>
        /// Beam search
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;Sequence[]&gt; representing the asynchronous operation.</returns>
        protected virtual async Task<Sequence[]> BeamSearchAsync(O options, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var sampler = GetSampler(options, true);
            var logitsProcessors = GetLogitsProcessor(options);
            var tokenProcessors = GetTokenProcessors(options);

            var initialPass = true;
            var progressTokens = new List<long>();
            var sequence = await InitializeAsync(options);
            var activeBeams = new SequenceCollection(sequence, options.Beams);
            while (!cancellationToken.IsCancellationRequested)
            {
                cancellationToken.ThrowIfCancellationRequested();

                var beamCandidates = new SequenceCollection();
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
                    var samples = initialPass
                        ? sampler.Sample(logits, Math.Max(options.Beams, options.TopK), 1, options.Temperature)
                        : sampler.Sample(logits, options.TopK, options.TopP, options.Temperature);
                    foreach (var sample in samples)
                    {
                        cancellationToken.ThrowIfCancellationRequested();

                        var beamCandidate = beam.Clone();
                        beamCandidate.Tokens.Add(sample.TokenId);
                        beamCandidate.Score += sample.Score;
                        beamCandidate.PenaltyScore = GetLengthPenalty(beamCandidate, options.LengthPenalty);
                        beamCandidates.Add(beamCandidate);
                    }

                    initialPass = false;
                }

                // Clear old active candidates
                activeBeams.Remove(beamCandidates);
                activeBeams.Clear();

                // Set new active candidates
                foreach (var candidate in GetSequenceCandidates(beamCandidates, options))
                {
                    activeBeams.Add(candidate);
                    beamCandidates.Remove(candidate);
                }
                beamCandidates.Clear();


                // Process Beams
                var bestBeam = activeBeams[0];
                foreach (var beam in activeBeams)
                {
                    if (beam.IsComplete)
                        continue;

                    if (tokenProcessors.Any(x => x.Process(beam)))
                    {
                        beam.IsComplete = true;
                    }
                }

                // Notify
                NotifyProgress(progressCallback, bestBeam, progressTokens);

                // Check Completion
                if (activeBeams.All(x => x.IsComplete))
                    break;

                // Check Early Stopping
                if (IsEarlyStopping(activeBeams, options))
                    break;
            }

            // Return beam reuslts
            return NormalizeAndSort(activeBeams, options);
        }


        /// <summary>
        /// Gets the next sequence candidates.
        /// </summary>
        /// <param name="candidates">The sequences.</param>
        /// <param name="options">The options.</param>
        protected virtual IEnumerable<Sequence> GetSequenceCandidates(SequenceCollection candidates, O options)
        {
            // TODO: Diversity Penalty
            _sequenceComparer.SetLength(options.DiversityLength);

            // Select new cadidates
            var completed = candidates.Where(x => x.IsValid && x.IsComplete);
            var newcandidates = candidates
                .Where(x => !x.IsComplete)
                .GroupBy(g => g, _sequenceComparer)
                .Select(g => g.OrderByDescending(s => s.Score).First())
                .OrderByDescending(s => s.PenaltyScore)
                .Take(options.Beams);

            return completed
                .Concat(newcandidates)
                .OrderByDescending(s => s.PenaltyScore);
        }


        /// <summary>
        /// Determines whether early stop is required.
        /// </summary>
        /// <param name="sequences">The sequences.</param>
        /// <param name="options">The options.</param>
        protected virtual bool IsEarlyStopping(SequenceCollection sequences, O options)
        {
            if (options.EarlyStopping != EarlyStopping.None)
            {
                var finished = sequences.Where(x => x.IsComplete);
                if (options.EarlyStopping == EarlyStopping.BestBeam)
                {
                    var unfinished = sequences.Where(x => !x.IsComplete);
                    if (finished.Any())
                    {
                        var bestFinished = finished.Max(b => b.PenaltyScore);
                        var bestUnfinished = unfinished.Max(b => b.PenaltyScore);
                        if (bestFinished >= bestUnfinished)
                            return true;
                    }
                }
                else if (options.EarlyStopping == EarlyStopping.BeamCount)
                {
                    if (finished.Count() >= options.Beams)
                        return true;
                }
            }
            return false;
        }


        /// <summary>
        /// Gets the length penalty.
        /// </summary>
        /// <param name="sequence">The sequence.</param>
        /// <param name="penalty">The penalty.</param>
        protected virtual float GetLengthPenalty(Sequence sequence, float penalty)
        {
            return sequence.Score / MathF.Pow((5.0f + sequence.Length) / 6.0f, penalty);
        }


        /// <summary>
        /// Normalizes the and sort.
        /// </summary>
        /// <param name="sequences">The sequences.</param>
        /// <param name="options">The options.</param>
        protected virtual Sequence[] NormalizeAndSort(SequenceCollection sequences, O options)
        {
            var resultSequences = sequences
                .Where(x => x.IsComplete)
                .OrderByDescending(s => s.PenaltyScore)
                .ToArray();

            sequences.Remove(resultSequences);
            sequences.Clear();
            return resultSequences;
        }


        /// <summary>
        /// Gets the PositionIds.
        /// </summary>
        /// <param name="metadata">The metadata.</param>
        /// <param name="startPosition">The start position.</param>
        /// <param name="endPosition">The end position.</param>
        /// <returns>Tensor&lt;System.Int64&gt;.</returns>
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
        /// Notify token progress.
        /// </summary>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="sequence">The sequence.</param>
        /// <param name="previousTokens">The previous tokens.</param>
        protected void NotifyProgress(IProgress<GenerateProgress> progressCallback, Sequence sequence, List<long> previousTokens = null)
        {
            if (progressCallback == null)
                return;

            string result;
            var hasBeamChanged = false;
            var newToken = sequence.Tokens[^1];
            if (previousTokens == null)
            {
                result = Tokenizer.Decode(newToken);
            }
            else
            {
                var newTokens = sequence.Tokens[..^1];
                if (sequence.Length == previousTokens.Count)
                    return;

                hasBeamChanged = !previousTokens.SequenceEqual(newTokens);
                if (hasBeamChanged)
                {
                    previousTokens.Clear();
                    previousTokens.AddRange(sequence.Tokens);
                    result = Tokenizer.Decode(previousTokens);
                }
                else
                {
                    previousTokens.Add(newToken);
                    result = Tokenizer.Decode(newToken);
                }
            }

            progressCallback.Report(new GenerateProgress
            {
                Result = result,
                IsReset = hasBeamChanged
            });
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
}