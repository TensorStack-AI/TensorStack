// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.IO;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;
using TensorStack.TextGeneration.Common;
using TensorStack.TextGeneration.Pipelines.Florence;
using TensorStack.TextGeneration.Processing;
using TensorStack.TextGeneration.Processing.Sampler;
using TensorStack.TextGeneration.Tokenizers;

namespace TensorStack.TextGeneration.Pipelines.Whisper
{
    public class WhisperPipeline : EncoderDecoderPipeline, ITextGeneration
    {
        private readonly WhisperPreprocessor _preProcessor;
        private Tensor<float> _audioSample;

        /// <summary>
        /// Initializes a new instance of the <see cref="WhisperPipeline"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public WhisperPipeline(WhisperConfig configuration)
            : base(configuration)
        {
            _preProcessor = new WhisperPreprocessor();
        }


        /// <summary>
        /// Runs the GreedySearch inference
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;GenerateResult&gt; representing the asynchronous operation.</returns>
        public async Task<GenerateResult> RunAsync(GenerateOptions options, IProgress<RunProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            await TokenizePromptAsync(options);

            var sequence = await GreedySearchAsync(options, cancellationToken);
            using (sequence)
            {
                return new GenerateResult
                {
                    Score = sequence.Score,
                    PenaltyScore = sequence.PenaltyScore,
                    Result = Tokenizer.Decode(sequence.Tokens)
                };
            }
        }


        /// <summary>
        /// Runs the BeamSearch inference
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        public async Task<GenerateResult[]> RunAsync(SearchOptions options, IProgress<RunProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            await TokenizePromptAsync(options);

            var sequences = await BeamSearchAsync(options, cancellationToken);
            var results = new GenerateResult[sequences.Length];
            for (int i = 0; i < sequences.Length; i++)
            {
                var sequence = sequences[i];
                using (sequence)
                {
                    results[i] = new GenerateResult
                    {
                        Beam = sequence.Id,
                        Score = sequence.Score,
                        PenaltyScore = sequence.PenaltyScore,
                        Result = Tokenizer.Decode(sequence.Tokens)
                    };
                }
            }
            return results;
        }


        /// <summary>
        /// Tokenize the prompt
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns>A Task representing the asynchronous operation.</returns>
        protected override async Task TokenizePromptAsync(GenerateOptions options)
        {
            _audioSample = _preProcessor.Process(options.Prompt); // TODO: Batch [1, 80, 3000]
            EncoderOutput = await RunEncoderAsync();
        }


        /// <summary>
        /// Initialize decoder cache
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns>A Task&lt;Sequence&gt; representing the asynchronous operation.</returns>
        protected override async Task<Sequence> InitializeAsync(GenerateOptions options)
        {
            var modelMetadata = await Decoder.LoadAsync();
            var dataType = modelMetadata.Outputs[0].Value.ElementDataType;
            var kvCache = new KVCacheEncoderDecoder(dataType, DecoderConfig.NumHeads, DecoderConfig.NumLayers, DecoderConfig.HiddenSize);
            var sequence = new Sequence(kvCache, 50258); // BOS
            sequence.Tokens.Add(50259); // Language (en)
            sequence.Tokens.Add(50359); // TaskType (Transcribe)
            return sequence;
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
                parameters.AddInput(_audioSample);
                parameters.AddOutput([1, 1500, 512]); //last_hidden_state
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
            var modelMetadata = await Decoder.LoadAsync();
            var useCacheBranch = sequence.Initialize(sequence.Length);
            var inputIds = useCacheBranch
                ? new Tensor<long>(new long[] { sequence.Tokens[^1] }, [1, 1])
                : new Tensor<long>(sequence.Tokens.ToArray(), [1, sequence.Length]);
            using (var parameters = new ModelParameters(modelMetadata))
            {
                // Inputs
                parameters.AddInput(inputIds);
                parameters.AddInput(EncoderOutput);
                foreach (var pastKeyValue in sequence.Cache)
                    parameters.AddInput(pastKeyValue, false);
                parameters.AddScalarInput(useCacheBranch);

                // Outputs
                foreach (var output in modelMetadata.Outputs)
                    parameters.AddOutput();

                // Result
                var modelResult = Decoder.RunInference(parameters);
                using (var logitsResult = modelResult[0])
                {
                    var logits = logitsResult.ToTensor();
                    var presentKeyValues = modelResult.Skip(1).Take(Configuration.DecoderConfig.NumLayers * 4).ToArray();

                    logits = useCacheBranch
                      ? logits.Reshape([logits.Dimensions[0], logits.Dimensions[2]])
                      : logits.Reshape([logits.Dimensions[1], logits.Dimensions[2]]).Split().LastOrDefault();

                    sequence.UpdateCache(presentKeyValues, useCacheBranch);
                    return logits;
                }
            }
        }


        /// <summary>
        /// Gets the sampler.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="isBeamSerach">if set to <c>true</c> [is beam serach].</param>
        /// <returns>Sampler.</returns>
        protected override Sampler GetSampler(GenerateOptions options, bool isBeamSerach)
        {
            return new GreedySampler(options);
        }


        /// <summary>
        /// Creates the Summary Pipeline
        /// </summary>
        /// <param name="provider">The provider.</param>
        /// <param name="modelPath">The model path.</param>
        /// <param name="tokenizerModel">The tokenizer model.</param>
        /// <param name="decoderModel">The decoder model.</param>
        /// <param name="encoderModel">The encoder model.</param>
        /// <returns>SummaryPipeline.</returns>
        public static WhisperPipeline Create(ExecutionProvider provider, string modelPath, string decoderModel = "decoder_model_merged.onnx", string encoderModel = "encoder_model.onnx")
        {
            return Create(provider, provider, modelPath, decoderModel, encoderModel);
        }


        /// <summary>
        /// Creates the Summary Pipeline
        /// </summary>
        /// <param name="encoderProvider">The encoder provider.</param>
        /// <param name="decoderProvider">The decoder provider.</param>
        /// <param name="modelPath">The model path.</param>
        /// <param name="tokenizerModel">The tokenizer model.</param>
        /// <param name="decoderModel">The decoder model.</param>
        /// <param name="encoderModel">The encoder model.</param>
        /// <returns>SummaryPipeline.</returns>
        public static WhisperPipeline Create(ExecutionProvider encoderProvider, ExecutionProvider decoderProvider, string modelPath, string decoderModel = "decoder_model_merged.onnx", string encoderModel = "encoder_model.onnx")
        {
            var config = new WhisperConfig
            {
                Tokenizer = new WhisperTokenizer(new TokenizerConfig
                {
                    BOS = 50257,
                    EOS = 50257,
                    Path = modelPath
                }),
                EncoderConfig = new EncoderConfig
                {
                    Path = Path.Combine(modelPath, encoderModel),
                    VocabSize = 51865,
                    NumHeads = 8,
                    NumLayers = 6,
                    NumKVHeads = 8,
                    HiddenSize = 512,
                },
                DecoderConfig = new DecoderConfig
                {
                    Path = Path.Combine(modelPath, decoderModel),
                    VocabSize = 51865,
                    NumHeads = 8,
                    NumLayers = 6,
                    NumKVHeads = 8,
                    HiddenSize = 512,
                }
            };

            config.EncoderConfig.SetProvider(encoderProvider);
            config.DecoderConfig.SetProvider(decoderProvider);
            return new WhisperPipeline(config);
        }
    }
}