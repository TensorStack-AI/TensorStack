// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.Extensions.Logging;
using System;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Pipeline;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Common;
using TensorStack.StableDiffusion.Enums;
using TensorStack.StableDiffusion.Models;
using TensorStack.TextGeneration.Tokenizers;

namespace TensorStack.StableDiffusion.Pipelines.StableCascade
{
    public class StableCascadePipeline : StableCascadeBase, IPipeline<ImageTensor, GenerateOptions, GenerateProgress>
    {
        public StableCascadePipeline(StableCascadeUNet priorUent, StableCascadeUNet decoderUnet, CLIPTokenizer tokenizer, CLIPTextModelWithProjection textEncoder, PaellaVQModel imageDecoder, CLIPVisionModelWithProjection imageEncoder, ILogger logger = null)
            : base(priorUent, decoderUnet, tokenizer, textEncoder, imageDecoder, imageEncoder, logger) { }

        public StableCascadePipeline(StableCascadeConfig configuration, ILogger logger = null)
            : base(configuration, logger) { }


        /// <summary>
        /// Run ImageTensor pipeline.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task<ImageTensor> RunAsync(GenerateOptions options, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            ValidateOptions(options);
            await CheckPipelineState(options);

            var prompt = await CreatePromptAsync(options, cancellationToken);

            // Prior Unet
            var priorPerformGuidance = options.GuidanceScale > 0;
            var priorPromptEmbeds = prompt.GetPromptEmbeds(priorPerformGuidance);
            var priorPooledPromptEmbeds = prompt.GetPromptPooledEmbeds(priorPerformGuidance);
            var priorLatents = await RunPriorAsync(options, priorPromptEmbeds, priorPooledPromptEmbeds, priorPerformGuidance, progressCallback, cancellationToken);

            // Decoder Unet
            var decodeSchedulerOptions = options with
            {
                Steps = options.Steps2,
                GuidanceScale = options.GuidanceScale2
            };
            var decoderPerformGuidance = decodeSchedulerOptions.GuidanceScale > 0;
            var decoderPromptEmbeds = prompt.GetPromptEmbeds(decoderPerformGuidance);
            var decoderPooledPromptEmbeds = prompt.GetPromptPooledEmbeds(decoderPerformGuidance);
            var decoderLatents = await RunDecoderAsync(decodeSchedulerOptions, priorLatents, decoderPromptEmbeds, decoderPooledPromptEmbeds, decoderPerformGuidance, progressCallback, cancellationToken);

            // Decode Latents
            return await DecodeLatentsAsync(options, decoderLatents, cancellationToken);
        }


        /// <summary>
        /// Create StableCascade pipeline from StableCascadeConfig file
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>StableCascadePipeline.</returns>
        public static StableCascadePipeline FromConfig(string configFile, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new StableCascadePipeline(StableCascadeConfig.FromFile(configFile, executionProvider), logger);
        }


        /// <summary>
        /// Create StableCascade pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>StableCascadePipeline.</returns>
        public static StableCascadePipeline FromFolder(string modelFolder, ModelType modelType, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new StableCascadePipeline(StableCascadeConfig.FromFolder(modelFolder, modelType, executionProvider), logger);
        }


        /// <summary>
        /// Create StableCascade pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>StableCascadePipeline.</returns>
        public static StableCascadePipeline FromFolder(string modelFolder, string variant, ModelType modelType, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new StableCascadePipeline(StableCascadeConfig.FromFolder(modelFolder, variant, modelType, executionProvider), logger);
        }
    }
}
