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

namespace TensorStack.StableDiffusion.Pipelines.StableDiffusion
{
    public class StableDiffusionVideoPipeline : StableDiffusionBase, IPipeline<VideoTensor, GenerateOptions, GenerateProgress>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionVideoPipeline"/> class.
        /// </summary>
        /// <param name="unet">The unet.</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="autoEncoder">The automatic encoder.</param>
        /// <param name="logger">The logger.</param>
        public StableDiffusionVideoPipeline(UNetConditionalModel unet, CLIPTokenizer tokenizer, CLIPTextModel textEncoder, AutoEncoderModel autoEncoder, ILogger logger = null)
            : base(unet, tokenizer, textEncoder, autoEncoder, logger) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionVideoPipeline"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="logger">The logger.</param>
        public StableDiffusionVideoPipeline(StableDiffusionConfig configuration, ILogger logger = null)
            : base(configuration, logger) { }


        protected override void ValidateOptions(GenerateOptions options)
        {
            base.ValidateOptions(options);
            if (options.InputVideo is null)
                throw new ArgumentException("InputVideo is null");
        }


        /// <summary>
        /// Run VideoTensor pipeline.
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="progressCallback">The progress callback.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task<VideoTensor> RunAsync(GenerateOptions options, IProgress<GenerateProgress> progressCallback = null, CancellationToken cancellationToken = default)
        {
            ValidateOptions(options);
            await CheckPipelineState(options);

            var prompt = await CreatePromptAsync(options, cancellationToken);
            using (var scheduler = CreateScheduler(options))
            {
                var latents = !options.HasControlNet
                    ? await RunInferenceAsync(options, scheduler, prompt, progressCallback, cancellationToken)
                    : await RunInferenceAsync(options, options.ControlNet, scheduler, prompt, progressCallback, cancellationToken);
                return default;// await DecodeLatentsAsync(options, latents, cancellationToken);
            }
        }


        /// <summary>
        /// Create StableDiffusion pipeline from StableDiffusionConfig file
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>StableDiffusionVideoPipeline.</returns>
        public static StableDiffusionVideoPipeline FromConfig(string configFile, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new StableDiffusionVideoPipeline(StableDiffusionConfig.FromFile(configFile, executionProvider), logger);
        }


        /// <summary>
        /// Create StableDiffusion pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>StableDiffusionVideoPipeline.</returns>
        public static StableDiffusionVideoPipeline FromFolder(string modelFolder, ModelType modelType, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new StableDiffusionVideoPipeline(StableDiffusionConfig.FromFolder(modelFolder, modelType, executionProvider), logger);
        }


        /// <summary>
        /// Create StableDiffusion pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>StableDiffusionVideoPipeline.</returns>
        public static StableDiffusionVideoPipeline FromFolder(string modelFolder, string variant, ModelType modelType, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new StableDiffusionVideoPipeline(StableDiffusionConfig.FromFolder(modelFolder, variant, modelType, executionProvider), logger);
        }


        /// <summary>
        /// Create StableDiffusion pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>StableDiffusionVideoPipeline.</returns>
        public static StableDiffusionVideoPipeline FromFolder(string modelFolder, string variant, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new StableDiffusionVideoPipeline(StableDiffusionConfig.FromFolder(modelFolder, variant, executionProvider), logger);
        }
    }
}
