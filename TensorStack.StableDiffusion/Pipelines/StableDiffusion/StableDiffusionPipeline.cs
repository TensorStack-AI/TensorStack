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
using TensorStack.StableDiffusion.Tokenizers;

namespace TensorStack.StableDiffusion.Pipelines.StableDiffusion
{
    public class StableDiffusionPipeline : StableDiffusionBase, IPipeline<ImageTensor, GenerateOptions, GenerateProgress>
    {
        public StableDiffusionPipeline(UNetConditionalModel unet, CLIPTokenizer tokenizer, CLIPTextModel textEncoder, AutoEncoderModel autoEncoder, ILogger logger = null)
            : base(unet, tokenizer, textEncoder, autoEncoder, logger) { }

        public StableDiffusionPipeline(StableDiffusionConfig configuration, ILogger logger = null)
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

            var prompt = await CreatePromptAsync(options, cancellationToken);
            using (var scheduler = CreateScheduler(options))
            {
                var latents = !options.HasControlNet
                    ? await RunInferenceAsync(options, scheduler, prompt, progressCallback, cancellationToken)
                    : await RunInferenceAsync(options, options.ControlNet, scheduler, prompt, progressCallback, cancellationToken);
                return await DecodeLatentsAsync(options, latents, cancellationToken);
            }
        }


        /// <summary>
        /// Create StableDiffusion pipeline from StableDiffusionConfig file
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>StableDiffusionPipeline.</returns>
        public static StableDiffusionPipeline FromConfig(string configFile, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new StableDiffusionPipeline(StableDiffusionConfig.FromFile(configFile, executionProvider), logger);
        }


        /// <summary>
        /// Create StableDiffusion pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>StableDiffusionPipeline.</returns>
        public static StableDiffusionPipeline FromFolder(string modelFolder, ModelType modelType, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new StableDiffusionPipeline(StableDiffusionConfig.FromFolder(modelFolder, modelType, executionProvider), logger);
        }
    }
}
