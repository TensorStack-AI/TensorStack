// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;
using TensorStack.TextGeneration.Common;
using TensorStack.TextGeneration.Processing;

namespace TensorStack.TextGeneration.Pipelines
{
    public abstract class EncoderDecoderPipeline : DecoderPipeline,
        IPipeline<GenerateResult, GenerateOptions>,
        IPipelineStream<GenerateResult, SearchOptions>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="EncoderDecoderPipeline"/> class.
        /// </summary>
        /// <param name="config">The configuration.</param>
        /// <param name="tokenizerConfig">The tokenizer configuration.</param>
        /// <param name="encoderConfig">The encoder configuration.</param>
        /// <param name="decoderConfig">The decoder configuration.</param>
        public EncoderDecoderPipeline(TransformerConfig configuration)
            : base(configuration.Tokenizer, configuration.DecoderConfig)
        {
            Configuration = configuration;
            Encoder = new ModelSession(EncoderConfig);
        }

        public TransformerConfig Configuration { get; }
        protected ModelSession Encoder { get; }
        protected EncoderConfig EncoderConfig => Configuration.EncoderConfig;
        protected Tensor<float> EncoderOutput { get; set; }


        /// <summary>
        /// Loads the models.
        /// </summary>
        public override async Task LoadAsync(CancellationToken cancellationToken = default)
        {
            await base.LoadAsync(cancellationToken);
            await Encoder.LoadAsync(cancellationToken: cancellationToken);
        }


        /// <summary>
        /// Unloads the models.
        /// </summary>
        public override async Task UnloadAsync(CancellationToken cancellationToken = default)
        {
            await base.UnloadAsync(cancellationToken: cancellationToken);
            await Encoder.UnloadAsync();
        }

        /// <summary>
        /// Run pipeline GreedySearch
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;GenerateResult&gt; representing the asynchronous operation.</returns>
        public virtual async Task<GenerateResult> RunAsync(GenerateOptions options, IProgress<RunProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            await TokenizePromptAsync(options);

            var sequence = await GreedySearchAsync(options, cancellationToken);
            using (sequence)
            {
                return new GenerateResult
                {
                    Score = sequence.Score,
                    Result = Tokenizer.Decode(sequence.Tokens)
                };
            }
        }


        /// <summary>
        /// Run pipeline BeamSearch
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;IAsyncEnumerable`1&gt; representing the asynchronous operation.</returns>
        public virtual async IAsyncEnumerable<GenerateResult> RunAsync(SearchOptions options, IProgress<RunProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            await TokenizePromptAsync(options);

            var sequences = await BeamSearchAsync(options, cancellationToken);
            foreach (var sequence in sequences)
            {
                using (sequence)
                {
                    yield return new GenerateResult
                    {
                        Beam = sequence.Id,
                        Score = sequence.Score,
                        Result = Tokenizer.Decode(sequence.Tokens)
                    };
                }
            }
        }


        /// <summary>
        /// Tokenize the prompt
        /// </summary>
        /// <param name="options">The options.</param>
        /// <returns>A Task representing the asynchronous operation.</returns>
        protected override async Task TokenizePromptAsync(GenerateOptions options)
        {
            await base.TokenizePromptAsync(options);
            EncoderOutput = await RunEncoderAsync();
        }


        /// <summary>
        /// Run encoder model.
        /// </summary>
        /// <param name="tokenizerOutput">The tokenizer output.</param>
        protected virtual async Task<Tensor<float>> RunEncoderAsync()
        {
            var modelMetadata = await Encoder.LoadAsync();
            using (var parameters = new ModelParameters(modelMetadata))
            {
                parameters.AddInput(TokenizerOutput.InputIds);
                parameters.AddInput(TokenizerOutput.Mask);
                parameters.AddOutput([1, TokenizerOutput.Length, EncoderConfig.HiddenSize]);
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
        /// <param name="tokenizerOutput">The tokenizer output.</param>
        /// <param name="encoderOutput">The encoder output.</param>
        protected override async Task<Tensor<float>> RunDecoderAsync(Sequence sequence)
        {
            var modelMetadata = await Decoder.LoadAsync();
            var useCacheBranch = sequence.Initialize(TokenizerOutput.Length);
            var inputIds = new Tensor<long>(new long[] { sequence.Tokens[^1] }, [1, 1]);
            using (var parameters = new ModelParameters(modelMetadata))
            {
                // Inputs
                parameters.AddInput(TokenizerOutput.Mask);
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
                    var presentKeyValues = modelResult.ToArray()[1..];

                    sequence.UpdateCache(presentKeyValues, useCacheBranch);
                    return logits.Reshape([logits.Dimensions[0], logits.Dimensions[2]]);
                }
            }
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
            return new Sequence(kvCache, Tokenizer.BOS);
        }


        /// <summary>
        /// Releases unmanaged and - optionally - managed resources.
        /// </summary>
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                Encoder?.Dispose();
            }
            base.Dispose(disposing);
        }
    }
}