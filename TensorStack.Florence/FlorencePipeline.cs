// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
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
using TensorStack.Core;
using TensorStack.Core.Inference;
using TensorStack.Florence;
using TensorStack.Florence.Common;
using TensorStack.Florence.Processing;
using TensorStack.Florence.Sampler;
using TensorStack.Florence.Tokenizer;

namespace TensorStack.Florence
{
    public class FlorencePipeline : IPipeline
    {
        private readonly FlorenceConfig _configuration;
        private readonly PreProcessor _preProcessor;
        private readonly PostProcessor _postProcessor;
        private readonly ModelSession _modelEncoder;
        private readonly ModelSession _modelEmbeds;
        private readonly ModelSession _modelVision;
        private readonly ModelSession _modelDecoder;
        private readonly FlorenceTokenizer _tokenizer;


        /// <summary>
        /// Initializes a new instance of the <see cref="FlorencePipeline"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="tokenizerPath">The tokenizer path.</param>
        /// <param name="embedsConfig">The embeds configuration.</param>
        /// <param name="encoderConfig">The encoder configuration.</param>
        /// <param name="visionConfig">The vision configuration.</param>
        /// <param name="decoderConfig">The decoder configuration.</param>
        public FlorencePipeline(FlorenceConfig configuration, ModelConfig tokenizerConfig, ModelConfig embedsConfig, ModelConfig encoderConfig, ModelConfig visionConfig, ModelConfig decoderConfig)
        {
            _configuration = configuration;
            _tokenizer = new FlorenceTokenizer(tokenizerConfig);
            _modelEmbeds = new ModelSession(embedsConfig);
            _modelEncoder = new ModelSession(encoderConfig);
            _modelVision = new ModelSession(visionConfig);
            _modelDecoder = new ModelSession(decoderConfig);
            _preProcessor = new PreProcessor(_configuration);
            _postProcessor = new PostProcessor(_configuration, _tokenizer);
        }


        /// <summary>
        /// Loads the models.
        /// </summary>
        public async Task LoadAsync()
        {
            await Task.WhenAll(
                _modelEncoder.LoadAsync(),
                _modelEmbeds.LoadAsync(),
                _modelVision.LoadAsync(),
                _modelDecoder.LoadAsync()
            );
        }


        /// <summary>
        /// Unloads the models.
        /// </summary>
        public async Task UnloadAsync()
        {
            await Task.WhenAll(
              _modelEncoder.UnloadAsync(),
              _modelEmbeds.UnloadAsync(),
              _modelVision.UnloadAsync(),
              _modelDecoder.UnloadAsync()
            );
        }


        /// <summary>
        /// Performs application-defined tasks associated with freeing, releasing, or resetting unmanaged resources.
        /// </summary>
        public void Dispose()
        {
            _modelEncoder.Dispose();
            _modelEmbeds.Dispose();
            _modelVision.Dispose();
            _modelDecoder.Dispose();
        }


        /// <summary>
        /// Runs the model inference
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        public async IAsyncEnumerable<GenerateResult> RunAsync(GenerateOptions options, IProgress<RunProgress> progressCallback = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var prompt = _preProcessor.Process(options);

            var textOutput = _tokenizer.Encode(prompt.TaskPrompt);

            var embedsOutput = await RunEmbedModelAsync(textOutput.InputIds);

            var visionOutput = await RunVisionModelAsync(textOutput, embedsOutput, prompt.PixelValues);

            var encoderOutput = await RunEncoderModelAsync(visionOutput);

            await foreach (var beamOutput in RunDecoderLoopAsync(options, visionOutput, encoderOutput, cancellationToken))
            {
                var processedBeamOutput = _postProcessor.Process(options, beamOutput.Tokens);
                processedBeamOutput.Score = beamOutput.Score;
                processedBeamOutput.BeamIndex = beamOutput.Index;
                yield return processedBeamOutput;
            }
        }


        /// <summary>
        /// Runs the embed model.
        /// </summary>
        /// <param name="textInputs">The text inputs.</param>
        /// <returns></returns>
        private async Task<Tensor<float>> RunEmbedModelAsync(Tensor<long> textInputs)
        {
            var sessionEmbedMetadata = await _modelEmbeds.LoadAsync();
            using (var sessionEmbedParams = new InferenceParameters(sessionEmbedMetadata))
            {
                sessionEmbedParams.AddInput(textInputs.AsTensorSpan());
                sessionEmbedParams.AddOutput([textInputs.Dimensions[0], textInputs.Dimensions[1], _configuration.DecoderHiddenSize]);
                using (var results = await _modelEmbeds.RunInferenceAsync(sessionEmbedParams))
                {
                    return results[0].ToTensor();
                }
            }
        }


        /// <summary>
        /// Runs the vision model.
        /// </summary>
        /// <param name="textInputs">The text inputs.</param>
        /// <param name="textEmbeds">The inputs embeds.</param>
        /// <param name="pixelValues">The pixel values.</param>
        /// <returns></returns>
        private async Task<VisionResult> RunVisionModelAsync(TokenizerResult textOutput, Tensor<float> textEmbeds, ImageTensor pixelValues)
        {
            var sessionVisionMetadata = await _modelVision.LoadAsync();
            using (var sessionVisionParams = new InferenceParameters(sessionVisionMetadata))
            {
                sessionVisionParams.AddInput(pixelValues.GetChannels(3));
                sessionVisionParams.AddOutput([textEmbeds.Dimensions[0], _configuration.ImageSeqLength, textEmbeds.Dimensions[2]]);
                using (var results = await _modelVision.RunInferenceAsync(sessionVisionParams))
                {
                    var imageFeatures = results[0].ToTensor();
                    var ones = new  Tensor<long>(imageFeatures.Dimensions[..2], 1);
                    return new VisionResult
                    (
                        imageFeatures.Concatenate(textEmbeds, axis: 1),
                        ones.Concatenate(textOutput.Mask, axis: 1)
                    );
                }
            }
        }


        /// <summary>
        /// Runs the encoder model.
        /// </summary>
        /// <param name="inputEmbeds">The input embeds.</param>
        /// <returns></returns>
        private async Task<Tensor<float>> RunEncoderModelAsync(VisionResult visionOutput)
        {
            var sessionEncoderMetadata = await _modelEncoder.LoadAsync();
            using (var sessionEncoderParams = new InferenceParameters(sessionEncoderMetadata))
            {
                sessionEncoderParams.AddInput(visionOutput.Mask.AsTensorSpan());
                sessionEncoderParams.AddInput(visionOutput.Embeds.AsTensorSpan());
                sessionEncoderParams.AddOutput(visionOutput.Embeds.Dimensions);
                using (var results = await _modelEncoder.RunInferenceAsync(sessionEncoderParams))
                {
                    return results[0].ToTensor();
                }
            }
        }


        /// <summary>
        /// Runs the decoder model.
        /// </summary>
        /// <param name="inputEmbeds">The input embeds.</param>
        /// <param name="encoderOutput">The encoder output.</param>
        /// <param name="decoderInputEmbeds">The decoder input embeds.</param>
        /// <param name="pastKeyValueCache">The past key value cache.</param>
        /// <returns></returns>
        private async Task<Tensor<float>> RunDecoderModelAsync(VisionResult visionOutput, Tensor<float> encoderOutput, Tensor<float> decoderInputEmbeds, PastValueCache pastKeyValueCache)
        {
            var numBeams = decoderInputEmbeds.Dimensions[0];
            var useCacheBranch = pastKeyValueCache.IsInitialized;
            if (!useCacheBranch)
                pastKeyValueCache.Initialize(numBeams);

            var sessionDecoderMergedMetadata = await _modelDecoder.LoadAsync();
            using (var sessionDecoderMergedParams = new InferenceParameters(sessionDecoderMergedMetadata))
            {
                sessionDecoderMergedParams.AddInput(visionOutput.Mask.AsTensorSpan());
                sessionDecoderMergedParams.AddInput(encoderOutput.Repeat(numBeams).AsTensorSpan());
                sessionDecoderMergedParams.AddInput(decoderInputEmbeds.AsTensorSpan());
                foreach (var pastKeyValue in pastKeyValueCache.PastValues)
                    sessionDecoderMergedParams.AddInput(pastKeyValue.AsTensorSpan());
                sessionDecoderMergedParams.AddInput(new Tensor<bool>(new[] { useCacheBranch }, [1]).AsTensorSpan());

                // TODO: Calculate output buffer sizes to allow async execution
                sessionDecoderMergedParams.AddOutput();
                foreach (var pastKeyValue in pastKeyValueCache.PastValues)
                    sessionDecoderMergedParams.AddOutput();

                using (var sessionDecoderMergedResult = _modelDecoder.RunInference(sessionDecoderMergedParams))
                {
                    var logits = sessionDecoderMergedResult[0].ToTensor();
                    var presentKeyValues = sessionDecoderMergedResult.ToArray()[1..];

                    pastKeyValueCache.Update(presentKeyValues, useCacheBranch);
                    return logits.Reshape([logits.Dimensions[0], logits.Dimensions[2]]);
                }
            }
        }


        /// <summary>
        /// Runs the decoder loop.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="inputEmbeds">The input embeds.</param>
        /// <param name="encoderOutput">The encoder output.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns></returns>
        private async IAsyncEnumerable<BeamResult> RunDecoderLoopAsync(GenerateOptions options, VisionResult visionOutput, Tensor<float> encoderOutput, [EnumeratorCancellation] CancellationToken cancellationToken = default)
        {
            var beamSearchResults = new BeamResult[options.NumBeams];
            var beamSearchSampler = new BeamSearchSampler(options.TopK);

            var decoderInputIds = new Tensor<long>([options.NumBeams, 1]);
            var resultInputIds = new List<long>().Repeat(options.NumBeams);

            var logitsProcessors = new ILogitsProcessor[]
            {
                new BOSLogitsProcessor(_tokenizer.BeginningOfSequenceId),
                new NoRepeatNGramLogitsProcessor(options.NoRepeatNgramSize)
            };

            var tokenProcessors = new ITokenProcessor[]
            {
                new EOSTokenProcessor(_tokenizer.EndOfSequenceId),
                new MaxLengthTokenProcessor(options.MaxLength)
            };

            using (var pastKeyValueCache = new PastValueCache(_configuration))
            {
                while (!cancellationToken.IsCancellationRequested)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    var decoderInputEmbeds = await RunEmbedModelAsync(decoderInputIds);
                    var logitsResult = await RunDecoderModelAsync(visionOutput, encoderOutput, decoderInputEmbeds, pastKeyValueCache);

                    foreach (var logitsProcessor in logitsProcessors)
                    {
                        logitsProcessor.Process(resultInputIds, logitsResult);
                    }

                    var beamNum = 0;
                    var newInputIds = new long[options.NumBeams];
                    foreach (var sampledToken in beamSearchSampler.Sample(resultInputIds, logitsResult))
                    {
                        var sampledScore = sampledToken.Score;
                        var sampledTokenId = sampledToken.TokenId;

                        if (beamSearchResults[beamNum] is null)
                            beamSearchResults[beamNum] = new BeamResult(beamNum, resultInputIds[beamNum]);

                        if (beamSearchResults[beamNum].IsComplete)
                            sampledTokenId = _tokenizer.EndOfSequenceId;

                        newInputIds[beamNum] = sampledTokenId;
                        resultInputIds[beamNum].Add(sampledTokenId);
                        beamSearchResults[beamNum].Score += sampledScore;
                        beamNum++;
                    }

                    // Next Inputs
                    decoderInputIds = new Tensor<long>(newInputIds, [newInputIds.Length, 1]);

                    // Token Processors
                    foreach (var tokenProcessor in tokenProcessors)
                    {
                        var processComplete = tokenProcessor.Process(beamSearchResults);
                        for (var i = 0; i < beamSearchResults.Length; ++i)
                        {
                            if (processComplete[i])
                            {
                                if (beamSearchResults[i].IsComplete)
                                    continue;

                                //  Return finished beam result
                                beamSearchResults[i].IsComplete = true;
                                yield return beamSearchResults[i];
                            }
                        }
                    }

                    // Return if all beams are complete
                    if (beamSearchResults.All(beam => beam.IsComplete))
                        yield break;
                }
            }
        }


        /// <summary>
        /// Creates a FlorencePipeline with the specified configuration.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="provider">The provider.</param>
        /// <param name="deviceId">The device identifier.</param>
        /// <returns>FlorencePipeline.</returns>
        public static FlorencePipeline Create(FlorenceConfig configuration, Provider provider = Provider.DirectML, int deviceId = 0)
        {
            var modelTokenizerConfig = new ModelConfig(configuration.Path, provider, deviceId, false);
            var modelEmbedsConfig = new ModelConfig(Path.Combine(configuration.Path, "embed_tokens.onnx"), provider, deviceId, false);
            var modelEncoderConfig = new ModelConfig(Path.Combine(configuration.Path, "encoder_model.onnx"), provider, deviceId, false);
            var modelVisionConfig = new ModelConfig(Path.Combine(configuration.Path, "vision_encoder.onnx"), provider, deviceId, false);
            var modelDecoderConfig = new ModelConfig(Path.Combine(configuration.Path, "decoder_model_merged.onnx"), provider, deviceId, false) with
            {
                // TODO: DirectML just creates garbage output
                // CPU & CUDA are ok, need to investigate
                Provider = provider == Provider.DirectML 
                    ? Provider.CPU 
                    : provider
            };
            return new FlorencePipeline(configuration, modelTokenizerConfig, modelEmbedsConfig, modelEncoderConfig, modelVisionConfig, modelDecoderConfig);
        }
    }
}