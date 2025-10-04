// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.CompilerServices;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;
using TensorStack.TextGeneration.Common;
using TensorStack.TextGeneration.Processing;
using TensorStack.TextGeneration.Tokenizers;

namespace TensorStack.TextGeneration.Pipelines.Florence
{
    public class FlorencePipeline : EncoderDecoderPipeline<FlorenceOptions>,
         IPipeline<GenerateResult, FlorenceOptions, GenerateProgress>,
         IPipeline<GenerateResult[], FlorenceSearchOptions, GenerateProgress>
    {
        private readonly FlorenceConfig _configuration;
        private readonly PreProcessor _preProcessor;
        private readonly PostProcessor _postProcessor;
        private readonly ModelSession _modelEmbeds;
        private readonly ModelSession _modelVision;
        private VisionResult _visionOutput;

        /// <summary>
        /// Initializes a new instance of the <see cref="FlorencePipeline"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="tokenizerPath">The tokenizer path.</param>
        /// <param name="embedsConfig">The embeds configuration.</param>
        /// <param name="encoderConfig">The encoder configuration.</param>
        /// <param name="visionConfig">The vision configuration.</param>
        /// <param name="decoderConfig">The decoder configuration.</param>
        public FlorencePipeline(FlorenceConfig configuration)
            : base(configuration)
        {
            _configuration = configuration;
            _modelEmbeds = new ModelSession(configuration.EmbedsConfig);
            _modelVision = new ModelSession(configuration.VisionConfig);
            _preProcessor = new PreProcessor(_configuration);
            _postProcessor = new PostProcessor(_configuration, configuration.Tokenizer as FlorenceTokenizer);
        }


        /// <summary>
        /// Loads the models.
        /// </summary>
        public override async Task LoadAsync(CancellationToken cancellationToken = default)
        {
            await base.UnloadAsync(cancellationToken);
            await Task.WhenAll
            (
                _modelEmbeds.LoadAsync(cancellationToken: cancellationToken),
                _modelVision.LoadAsync(cancellationToken: cancellationToken)
            );
        }


        /// <summary>
        /// Unloads the models.
        /// </summary>
        public override async Task UnloadAsync(CancellationToken cancellationToken = default)
        {
            await base.UnloadAsync(cancellationToken);
            await Task.WhenAll
            (
                _modelEmbeds.UnloadAsync(),
                _modelVision.UnloadAsync()
            );
        }


        /// <summary>
        /// Run Florence pipeline
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        /// <returns>A Task&lt;GenerateResult&gt; representing the asynchronous operation.</returns>
        public virtual async Task<GenerateResult> RunAsync(FlorenceOptions options, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var textPrompt = _preProcessor.ProcessPrompt(options);
            var imagePrompt = _preProcessor.ProcessImage(options);

            TokenizerOutput = await Tokenizer.EncodeAsync(textPrompt);
            var embedsOutput = await RunTextEmbedAsync(TokenizerOutput.InputIds);
            _visionOutput = await RunVisionEncoderAsync(embedsOutput, imagePrompt);
            EncoderOutput = await RunEncoderAsync();

            var sequence = await GreedySearchAsync(options, progressCallback, cancellationToken);
            using (sequence)
            {
                var processedBeamOutput = _postProcessor.Process(options, sequence.Tokens);
                return new GenerateResult
                {
                    Score = sequence.Score,
                    Result = processedBeamOutput.Result,
                    CoordinateResults = processedBeamOutput.CoordinateResults
                };
            }
        }


        public virtual async Task<GenerateResult[]> RunAsync(FlorenceSearchOptions options, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            var textPrompt = _preProcessor.ProcessPrompt(options);
            var imagePrompt = _preProcessor.ProcessImage(options);

            TokenizerOutput = await Tokenizer.EncodeAsync(textPrompt);
            var embedsOutput = await RunTextEmbedAsync(TokenizerOutput.InputIds);
            _visionOutput = await RunVisionEncoderAsync(embedsOutput, imagePrompt);
            EncoderOutput = await RunEncoderAsync();

            var sequences = await BeamSearchAsync(options, progressCallback, cancellationToken);
            var results = new GenerateResult[sequences.Length];
            for (int beam = 0; beam < sequences.Length; beam++)
            {
                var sequence = sequences[beam];
                using (sequence)
                {
                    var processedBeamOutput = _postProcessor.Process(options, sequence.Tokens);
                    results[beam] = new GenerateResult
                    {
                        Beam = beam,
                        Score = sequence.Score,
                        PenaltyScore = sequence.PenaltyScore,
                        Result = processedBeamOutput.Result,
                        CoordinateResults = processedBeamOutput.CoordinateResults
                    };
                }
            }
            return results;
        }


        /// <summary>
        /// Run encoder model.
        /// </summary>
        /// <returns>A Task&lt;Tensor`1&gt; representing the asynchronous operation.</returns>
        protected override async Task<Tensor<float>> RunEncoderAsync()
        {
            var sessionEncoderMetadata = await Encoder.LoadAsync();
            using (var sessionEncoderParams = new ModelParameters(sessionEncoderMetadata))
            {
                sessionEncoderParams.AddInput(_visionOutput.Mask.AsTensorSpan());
                sessionEncoderParams.AddInput(_visionOutput.Embeds.AsTensorSpan());
                sessionEncoderParams.AddOutput(_visionOutput.Embeds.Dimensions);
                using (var results = await Encoder.RunInferenceAsync(sessionEncoderParams))
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
            var useCacheBranch = sequence.Initialize(TokenizerOutput.Length);
            var inputIds = new Tensor<long>(new long[] { sequence.Tokens[^1] }, [1, 1]);
            var inputEmbeds = await RunTextEmbedAsync(inputIds);
            using (var parameters = new ModelParameters(modelMetadata))
            {
                // Inputs
                parameters.AddInput(_visionOutput.Mask);
                parameters.AddInput(EncoderOutput);
                parameters.AddInput(inputEmbeds);
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
        /// Run text embed model
        /// </summary>
        /// <param name="inputIds">The input ids.</param>
        /// <returns>A Task&lt;Tensor`1&gt; representing the asynchronous operation.</returns>
        private async Task<Tensor<float>> RunTextEmbedAsync(Tensor<long> inputIds)
        {
            var sessionEmbedMetadata = await _modelEmbeds.LoadAsync();
            using (var sessionEmbedParams = new ModelParameters(sessionEmbedMetadata))
            {
                sessionEmbedParams.AddInput(inputIds.AsTensorSpan());
                sessionEmbedParams.AddOutput([inputIds.Dimensions[0], inputIds.Dimensions[1], _configuration.DecoderConfig.HiddenSize]);
                using (var results = await _modelEmbeds.RunInferenceAsync(sessionEmbedParams))
                {
                    return results[0].ToTensor();
                }
            }
        }


        /// <summary>
        /// Run vision encoder model
        /// </summary>
        /// <param name="inputEmbeds">The input embeds.</param>
        /// <param name="pixelValues">The pixel values.</param>
        /// <returns>A Task&lt;VisionResult&gt; representing the asynchronous operation.</returns>
        private async Task<VisionResult> RunVisionEncoderAsync(Tensor<float> inputEmbeds, ImageTensor pixelValues)
        {
            var sessionVisionMetadata = await _modelVision.LoadAsync();
            using (var sessionVisionParams = new ModelParameters(sessionVisionMetadata))
            {
                sessionVisionParams.AddInput(pixelValues.GetChannels(3));
                sessionVisionParams.AddOutput([inputEmbeds.Dimensions[0], _configuration.ImageSeqLength, inputEmbeds.Dimensions[2]]);
                using (var results = await _modelVision.RunInferenceAsync(sessionVisionParams))
                {
                    var imageFeatures = results[0].ToTensor();
                    var ones = new Tensor<long>(imageFeatures.Dimensions[..2], 1);
                    return new VisionResult
                    (
                        imageFeatures.Concatenate(inputEmbeds, axis: 1),
                        ones.Concatenate(TokenizerOutput.Mask, axis: 1)
                    );
                }
            }
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        protected override void Dispose(bool disposing)
        {
            if (disposing)
            {
                _modelEmbeds.Dispose();
                _modelVision.Dispose();
            }
            base.Dispose(disposing);
        }


        /// <summary>
        /// Creates a FlorencePipeline with the specified configuration.
        /// </summary>
        /// <param name="provider">The provider.</param>
        /// <param name="modelPath">The model path.</param>
        /// <param name="encoderModel">The encoder model.</param>
        /// <param name="decoderModel">The decoder model.</param>
        /// <param name="embedModel">The embed model.</param>
        /// <param name="visionModel">The vision model.</param>
        /// <returns>FlorencePipeline.</returns>
        public static FlorencePipeline Create(ExecutionProvider provider, string modelPath, FlorenceType modelType, string encoderModel = "encoder_model.onnx", string decoderModel = "decoder_model_merged.onnx", string embedModel = "embed_tokens.onnx", string visionModel = "vision_encoder.onnx")
        {
            return Create(provider, provider, provider, provider, modelPath, modelType, encoderModel, decoderModel, embedModel, visionModel);
        }


        /// <summary>
        /// Creates a FlorencePipeline with the specified configuration.
        /// </summary>
        /// <param name="encoderProvider">The encoder provider.</param>
        /// <param name="decoderProvider">The decoder provider.</param>
        /// <param name="embedsProvider">The embeds provider.</param>
        /// <param name="visionProvider">The vision provider.</param>
        /// <param name="modelPath">The model path.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="encoderModel">The encoder model.</param>
        /// <param name="decoderModel">The decoder model.</param>
        /// <param name="embedModel">The embed model.</param>
        /// <param name="visionModel">The vision model.</param>
        /// <returns>FlorencePipeline.</returns>
        public static FlorencePipeline Create(ExecutionProvider encoderProvider, ExecutionProvider decoderProvider, ExecutionProvider embedsProvider, ExecutionProvider visionProvider, string modelPath, FlorenceType modelType, string encoderModel = "encoder_model.onnx", string decoderModel = "decoder_model_merged.onnx", string embedModel = "embed_tokens.onnx", string visionModel = "vision_encoder.onnx")
        {
            var numLayers = 6;
            var numHeads = 12;
            var numKVHeads = 12;
            var hiddenSize = 768;
            var vocabSize = 51289;
            if (modelType == FlorenceType.Large)
            {
                numLayers = 12;
                numHeads = 16;
                numKVHeads = 16;
                hiddenSize = 1024;
            }

            var config = new FlorenceConfig
            {
                Tokenizer = new FlorenceTokenizer(new TokenizerConfig
                {
                    EOS = 2,
                    Path = modelPath
                }),
                EncoderConfig = new EncoderConfig
                {
                    NumLayers = numLayers,
                    NumHeads = numHeads,
                    NumKVHeads = numKVHeads,
                    HiddenSize = hiddenSize,
                    VocabSize = vocabSize,
                    Path = Path.Combine(modelPath, encoderModel),
                },
                DecoderConfig = new DecoderConfig
                {
                    NumLayers = numLayers,
                    NumHeads = numHeads,
                    NumKVHeads = numKVHeads,
                    HiddenSize = hiddenSize,
                    VocabSize = vocabSize,
                    Path = Path.Combine(modelPath, decoderModel),
                },
                EmbedsConfig = new ModelConfig
                {
                    Path = Path.Combine(modelPath, embedModel),
                },
                VisionConfig = new ModelConfig
                {
                    Path = Path.Combine(modelPath, visionModel),
                }
            };

            config.EncoderConfig.SetProvider(encoderProvider);
            config.DecoderConfig.SetProvider(decoderProvider);
            config.EmbedsConfig.SetProvider(embedsProvider);
            config.VisionConfig.SetProvider(visionProvider);
            return new FlorencePipeline(config);
        }
    }
}