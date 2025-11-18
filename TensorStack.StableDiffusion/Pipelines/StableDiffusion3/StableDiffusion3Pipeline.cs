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

namespace TensorStack.StableDiffusion.Pipelines.StableDiffusion3
{
    public class StableDiffusion3Pipeline : StableDiffusion3Base, IPipeline<ImageTensor, GenerateOptions, GenerateProgress>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusion3Pipeline"/> class.
        /// </summary>
        /// <param name="transformer">The transformer.</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="tokenizer2">The tokenizer2.</param>
        /// <param name="tokenizer3">The tokenizer3.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="textEncoder2">The text encoder2.</param>
        /// <param name="textEncoder3">The text encoder3.</param>
        /// <param name="autoEncoder">The automatic encoder.</param>
        /// <param name="logger">The logger.</param>
        public StableDiffusion3Pipeline(TransformerSD3Model transformer, CLIPTokenizer tokenizer, CLIPTokenizer tokenizer2, T5Tokenizer tokenizer3, CLIPTextModel textEncoder, CLIPTextModelWithProjection textEncoder2, T5EncoderModel textEncoder3, AutoEncoderModel autoEncoder, ILogger logger = null)
            : base(transformer, tokenizer, tokenizer2, tokenizer3, textEncoder, textEncoder2, textEncoder3, autoEncoder, logger) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusion3Pipeline"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="logger">The logger.</param>
        public StableDiffusion3Pipeline(StableDiffusion3Config configuration, ILogger logger = null)
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
        /// Create StableDiffusion3 pipeline from StableDiffusionConfig file
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>StableDiffusion3Pipeline.</returns>
        public static StableDiffusion3Pipeline FromConfig(string configFile, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new StableDiffusion3Pipeline(StableDiffusion3Config.FromFile(configFile, executionProvider), logger);
        }


        /// <summary>
        /// Create StableDiffusion3 pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>StableDiffusion3Pipeline.</returns>
        public static StableDiffusion3Pipeline FromFolder(string modelFolder, ModelType modelType, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new StableDiffusion3Pipeline(StableDiffusion3Config.FromFolder(modelFolder, modelType, executionProvider), logger);
        }


        /// <summary>
        /// Create StableDiffusion3 pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>StableDiffusion3Pipeline.</returns>
        public static StableDiffusion3Pipeline FromFolder(string modelFolder, string variant, ModelType modelType, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new StableDiffusion3Pipeline(StableDiffusion3Config.FromFolder(modelFolder, variant, modelType, executionProvider), logger);
        }


        /// <summary>
        /// Create StableDiffusion3 pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>StableDiffusion3Pipeline.</returns>
        public static StableDiffusion3Pipeline FromFolder(string modelFolder, string variant, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new StableDiffusion3Pipeline(StableDiffusion3Config.FromFolder(modelFolder, variant, executionProvider), logger);
        }
    }
}
