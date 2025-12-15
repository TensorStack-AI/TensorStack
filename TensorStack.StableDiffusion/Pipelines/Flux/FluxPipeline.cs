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

namespace TensorStack.StableDiffusion.Pipelines.Flux
{
    public class FluxPipeline : FluxBase, IPipeline<ImageTensor, GenerateOptions, GenerateProgress>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="FluxPipeline"/> class.
        /// </summary>
        /// <param name="transformer">The transformer.</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="tokenizer2">The tokenizer2.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="textEncoder2">The text encoder2.</param>
        /// <param name="autoEncoder">The automatic encoder.</param>
        /// <param name="logger">The logger.</param>
        public FluxPipeline(TransformerFluxModel transformer, CLIPTokenizer tokenizer, T5Tokenizer tokenizer2, CLIPTextModel textEncoder, T5EncoderModel textEncoder2, AutoEncoderModel autoEncoder, ILogger logger = null)
            : base(transformer, tokenizer, tokenizer2, textEncoder, textEncoder2, autoEncoder, logger) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="FluxPipeline"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="logger">The logger.</param>
        public FluxPipeline(FluxConfig configuration, ILogger logger = null)
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
            using (var scheduler = CreateScheduler(options))
            {
                var latents = await RunInferenceAsync(options, scheduler, prompt, progressCallback, cancellationToken);
                return await DecodeLatentsAsync(options, latents, cancellationToken);
            }
        }


        /// <summary>
        /// Create Flux pipeline from StableDiffusionConfig file
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>FluxPipeline.</returns>
        public static FluxPipeline FromConfig(string configFile, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new FluxPipeline(FluxConfig.FromFile(configFile, executionProvider), logger);
        }


        /// <summary>
        /// Create Flux pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>FluxPipeline.</returns>
        public static FluxPipeline FromFolder(string modelFolder, ModelType modelType, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new FluxPipeline(FluxConfig.FromFolder(modelFolder, modelType, executionProvider), logger);
        }


        /// <summary>
        /// Create Flux pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>FluxPipeline.</returns>
        public static FluxPipeline FromFolder(string modelFolder, string variant, ModelType modelType, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new FluxPipeline(FluxConfig.FromFolder(modelFolder, variant, modelType, executionProvider), logger);
        }


        /// <summary>
        /// Create Flux pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>FluxPipeline.</returns>
        public static FluxPipeline FromFolder(string modelFolder, string variant, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new FluxPipeline(FluxConfig.FromFolder(modelFolder, variant, executionProvider), logger);
        }
    }
}
