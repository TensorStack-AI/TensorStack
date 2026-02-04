// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;
using TensorStack.TextGeneration.Cache;
using TensorStack.TextGeneration.Common;
using TensorStack.TextGeneration.Pipelines.Florence;
using TensorStack.TextGeneration.Processing;
using TensorStack.TextGeneration.Tokenizers;

namespace TensorStack.TextGeneration.Pipelines.Whisper
{
    public class WhisperPipeline : EncoderDecoderPipeline<WhisperOptions>,
        IPipeline<GenerateResult, WhisperOptions, GenerateProgress>,
        IPipeline<GenerateResult[], WhisperSearchOptions, GenerateProgress>
    {
        private readonly PreProcessor _preProcessor;
        private Tensor<float> _currentAudioSample;

        /// <summary>
        /// Initializes a new instance of the <see cref="WhisperPipeline"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public WhisperPipeline(WhisperConfig configuration)
            : base(configuration)
        {
            _preProcessor = new PreProcessor(configuration.MelFiltersPath);
        }

        protected WhisperTokenizer WhisperTokenizer => Tokenizer as WhisperTokenizer;

        /// <summary>
        /// Runs the GreedySearch inference
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;GenerateResult&gt; representing the asynchronous operation.</returns>
        public async Task<GenerateResult> RunAsync(WhisperOptions options, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var result = default(GenerateResult);
            var audioSamples = _preProcessor.ProcessInput(options.AudioInput, options.ChunkSize);
            var progress = new ChunkedProgress(audioSamples.Count, options.MaxLength, progressCallback);
            foreach (var sample in audioSamples)
            {
                await RunEncoderAsync(sample);
                var sequence = await GreedySearchAsync(options, progress.ProgressCallback, cancellationToken);
                using (sequence)
                {
                    if (result != null)
                    {
                        result.Score += sequence.Score.ZeroIfInfinity();
                        result.PenaltyScore = sequence.PenaltyScore.ZeroIfInfinity();
                        result.Result += Tokenizer.Decode(sequence.Tokens);
                    }
                    else
                    {
                        result = new GenerateResult
                        {
                            Score = sequence.Score.ZeroIfInfinity(),
                            PenaltyScore = sequence.PenaltyScore.ZeroIfInfinity(),
                            Result = Tokenizer.Decode(sequence.Tokens)
                        };
                    }
                }
                progress.ChunkComplete();
            }
            return result;
        }


        /// <summary>
        /// Runs the BeamSearch inference
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        public async Task<GenerateResult[]> RunAsync(WhisperSearchOptions options, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var results = new List<GenerateResult>();
            var audioSamples = _preProcessor.ProcessInput(options.AudioInput, options.ChunkSize);
            var progress = new ChunkedProgress(audioSamples.Count, options.MaxLength, progressCallback);
            foreach (var sample in audioSamples)
            {
                await RunEncoderAsync(sample);
                var sequences = await BeamSearchAsync(options, progress.ProgressCallback, cancellationToken);
                for (int beam = 0; beam < sequences.Length; beam++)
                {
                    var sequence = sequences[beam];
                    using (sequence)
                    {
                        var existing = results.ElementAtOrDefault(beam);
                        if (existing != null)
                        {
                            existing.Score += sequence.Score.ZeroIfInfinity();
                            existing.PenaltyScore = sequence.PenaltyScore.ZeroIfInfinity();
                            existing.Result += Tokenizer.Decode(sequence.Tokens);
                        }
                        else
                        {
                            results.Add(new GenerateResult
                            {
                                Beam = beam,
                                Score = sequence.Score.ZeroIfInfinity(),
                                PenaltyScore = sequence.PenaltyScore.ZeroIfInfinity(),
                                Result = Tokenizer.Decode(sequence.Tokens)
                            });
                        }
                    }
                }
                progress.ChunkComplete();
            }
            return [.. results];
        }


        /// <summary>
        /// Gets the logits processors.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns>ILogitsProcessor[].</returns>
        protected override ILogitsProcessor[] GetLogitsProcessor(WhisperOptions options)
        {
            return [.. base.GetLogitsProcessor(options), new SupressTokenLogitsProcessor(WhisperTokenizer.SuppressTokens)];
        }


        /// <summary>
        /// Initialize decoder cache
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns>A Task&lt;Sequence&gt; representing the asynchronous operation.</returns>
        protected override async Task<Sequence> InitializeAsync(WhisperOptions options)
        {
            var modelMetadata = await Decoder.LoadAsync();
            var kvCache = new KVCacheEncoderDecoder(modelMetadata, DecoderConfig.NumHeads, DecoderConfig.NumLayers, DecoderConfig.HiddenSize);
            var sequence = new Sequence(kvCache, Tokenizer.BOS);    // <|startoftranscript|>
            sequence.Tokens.Add((int)options.Language);             // <|en|>
            sequence.Tokens.Add((int)options.Task);                 // <|transcribe|>
            sequence.Tokens.Add(WhisperTokenizer.NoCaptionsToken);  // <|nocaptions|>   TODO: Options boolean
            sequence.Tokens.Add(WhisperTokenizer.NoTimestampToken); // <|notimestamps|> TODO: Timestamp processing
            return sequence;
        }


        /// <summary>
        /// Run encoder for the specified audio sample
        /// </summary>
        /// <param name="audioSample">The audio sample.</param>
        /// <returns>A Task representing the asynchronous operation.</returns>
        private async Task RunEncoderAsync(Tensor<float> audioSample)
        {
            _currentAudioSample = audioSample;
            EncoderOutput = await RunEncoderAsync();
            _currentAudioSample = null;
        }


        /// <summary>
        /// Run encoder model.
        /// </summary>
        /// <returns>A Task&lt;Tensor`1&gt; representing the asynchronous operation.</returns>
        protected override async Task<Tensor<float>> RunEncoderAsync()
        {
            var modelMetadata = await Encoder.LoadAsync();
            using (var parameters = new ModelParameters(modelMetadata))
            {
                parameters.AddInput(_currentAudioSample);
                parameters.AddOutput([1, 1500, Configuration.EncoderConfig.HiddenSize]);
                using (var results = await Encoder.RunInferenceAsync(parameters))
                {
                    return results[0].ToTensor();
                }
            }
        }


        /// <summary>
        /// Run decoder model
        /// </summary>
        /// <param name="sequence">The sequence.</param>
        /// <returns>A Task&lt;Tensor`1&gt; representing the asynchronous operation.</returns>
        protected override async Task<Tensor<float>> RunDecoderAsync(Sequence sequence)
        {
            var metadata = await Decoder.LoadAsync();
            var useCacheBranch = sequence.Initialize(sequence.Length);
            var inputIds = useCacheBranch
                ? new Tensor<long>(new long[] { sequence.Tokens[^1] }, [1, 1])
                : new Tensor<long>(sequence.Tokens.ToArray(), [1, sequence.Length]);
            using (var parameters = new ModelParameters(metadata))
            {
                // Inputs
                parameters.AddInput(inputIds);
                parameters.AddInput(EncoderOutput);
                foreach (var pastKeyValue in sequence.Cache)
                    parameters.AddInput(pastKeyValue, false);
                parameters.AddScalarInput(useCacheBranch);

                // Outputs
                foreach (var output in metadata.Outputs)
                    parameters.AddOutput();

                // Result
                var modelResult = Decoder.RunInference(parameters);
                using (var logitsResult = modelResult[0])
                {
                    var dimension = logitsResult.GetDimensions();
                    var logits = logitsResult.ToTensor(dimension[1..]);
                    if (!useCacheBranch)
                    {
                        logits = logits.Split().LastOrDefault();
                        foreach (var suppressToken in WhisperTokenizer.BeginSuppressTokens)
                            logits[0, suppressToken] = float.MinValue;
                    }

                    var presentKeyValues = modelResult.Skip(1).Take(Configuration.DecoderConfig.NumLayers * 4).ToArray();
                    sequence.UpdateCache(presentKeyValues, useCacheBranch);
                    return logits;
                }
            }
        }


        /// <summary>
        /// Creates the Summary Pipeline
        /// </summary>
        /// <param name="provider">The provider.</param>
        /// <param name="modelPath">The model path.</param>
        /// <param name="modelType">The model model.</param>
        /// <param name="melFiltersPath">The melFilters Path.</param>
        /// <param name="decoderModel">The decoder model.</param>
        /// <param name="encoderModel">The encoder model.</param>
        /// <returns>SummaryPipeline.</returns>
        public static WhisperPipeline Create(ExecutionProvider provider, string modelPath, WhisperType modelType, string melFiltersPath = "mel_filters.npz", string decoderModel = "decoder_model_merged.onnx", string encoderModel = "encoder_model.onnx")
        {
            return Create(provider, provider, modelPath, modelType, melFiltersPath, decoderModel, encoderModel);
        }


        /// <summary>
        /// Creates the Summary Pipeline
        /// </summary>
        /// <param name="encoderProvider">The encoder provider.</param>
        /// <param name="decoderProvider">The decoder provider.</param>
        /// <param name="modelPath">The model path.</param>
        /// <param name="modelType">The model model.</param>
        /// <param name="melFiltersPath">The melFilters Path.</param>
        /// <param name="decoderModel">The decoder model.</param>
        /// <param name="encoderModel">The encoder model.</param>
        /// <returns>SummaryPipeline.</returns>
        public static WhisperPipeline Create(ExecutionProvider encoderProvider, ExecutionProvider decoderProvider, string modelPath, WhisperType modelType, string melFiltersPath = "mel_filters.npz", string decoderModel = "decoder_model_merged.onnx", string encoderModel = "encoder_model.onnx")
        {
            var numHeads = 8;
            var numLayers = 6;
            var numKVHeads = 8;
            var hiddenSize = 512;
            if (modelType == WhisperType.Tiny)
            {
                numHeads = 6;
                numLayers = 4;
                numKVHeads = 6;
                hiddenSize = 384;
            }
            else if (modelType == WhisperType.Small)
            {
                numHeads = 12;
                numLayers = 12;
                numKVHeads = 12;
                hiddenSize = 768;
            }
            else if (modelType == WhisperType.Medium)
            {
                numHeads = 16;
                numLayers = 24;
                numKVHeads = 16;
                hiddenSize = 1024;
            }
            else if (modelType == WhisperType.Large)
            {
                numHeads = 20;
                numLayers = 32;
                numKVHeads = 20;
                hiddenSize = 1280;
            }

            var config = new WhisperConfig
            {
                MelFiltersPath = Path.Combine(modelPath, melFiltersPath),
                Tokenizer = new WhisperTokenizer(new TokenizerConfig
                {
                    BOS = 50258,
                    EOS = 50257,
                    Path = modelPath
                }),
                EncoderConfig = new EncoderConfig
                {
                    Path = Path.Combine(modelPath, encoderModel),
                    VocabSize = 51865,
                    NumHeads = numHeads,
                    NumLayers = numLayers,
                    NumKVHeads = numKVHeads,
                    HiddenSize = hiddenSize,
                },
                DecoderConfig = new DecoderConfig
                {
                    Path = Path.Combine(modelPath, decoderModel),
                    VocabSize = 51865,
                    NumHeads = numHeads,
                    NumLayers = numLayers,
                    NumKVHeads = numKVHeads,
                    HiddenSize = hiddenSize,
                }
            };

            config.EncoderConfig.SetProvider(encoderProvider);
            config.DecoderConfig.SetProvider(decoderProvider);
            return new WhisperPipeline(config);
        }
    }


    public class ChunkedProgress
    {
        private string _progressText;
        private string _totalProgressText;
        private Progress<GenerateProgress> _relayProgressCallback;

        public ChunkedProgress(int chunks, int maxlength, IProgress<GenerateProgress> progressCallback)
        {
            var progressValue = 0;
            var progressTotal = maxlength * chunks;
            _relayProgressCallback = new Progress<GenerateProgress>(progress =>
            {
                _progressText += progress.Result;
                if (progress.IsReset)
                    _progressText = progress.Result;

                progressValue++;
                progressCallback?.Report(new GenerateProgress
                {
                    IsReset = progress.IsReset,
                    Result = progress.IsReset ? _totalProgressText + _progressText : progress.Result,
                    Value = progressValue,
                    Maximum = progressTotal
                });

                //Debug.WriteLine(_totalProgressText + _progressText);
            });
        }

        public Progress<GenerateProgress> ProgressCallback => _relayProgressCallback;

        public void ChunkComplete()
        {
            _totalProgressText += _progressText;
            _progressText = string.Empty;
        }
    }
}