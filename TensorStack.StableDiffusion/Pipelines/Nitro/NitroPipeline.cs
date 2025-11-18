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
using TensorStack.TextGeneration.Pipelines.Llama;

namespace TensorStack.StableDiffusion.Pipelines.Nitro
{
    public class NitroPipeline : NitroBase, IPipeline<ImageTensor, GenerateOptions, GenerateProgress>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="NitroPipeline" /> class.
        /// </summary>
        /// <param name="transformer">The transformer.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="autoEncoder">The automatic encoder.</param>
        /// <param name="logger">The logger.</param>
        public NitroPipeline(TransformerNitroModel transformer, LlamaPipeline textEncoder, AutoEncoderModel autoEncoder, int outputSize, ILogger logger = null)
            : base(transformer, textEncoder, autoEncoder, outputSize, logger) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="NitroPipeline"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="logger">The logger.</param>
        public NitroPipeline(NitroConfig configuration, ILogger logger = null)
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
                var latents = await RunInferenceAsync(options, scheduler, prompt, progressCallback, cancellationToken);
                return await DecodeLatentsAsync(options, latents, cancellationToken);
            }
        }


        /// <summary>
        /// Create Nitro pipeline from StableDiffusionConfig file
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>NitroPipeline.</returns>
        public static NitroPipeline FromConfig(string configFile, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new NitroPipeline(NitroConfig.FromFile(configFile, executionProvider), logger);
        }


        /// <summary>
        /// Create Nitro pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="outputSize">Size of the output. [512, 1024]</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        public static NitroPipeline FromFolder(string modelFolder, int outputSize, ModelType modelType, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new NitroPipeline(NitroConfig.FromFolder(modelFolder, outputSize, modelType, executionProvider), logger);
        }


        /// <summary>
        /// Create Nitro pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="outputSize">Size of the output.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>NitroPipeline.</returns>
        public static NitroPipeline FromFolder(string modelFolder, string variant, int outputSize, ModelType modelType, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new NitroPipeline(NitroConfig.FromFolder(modelFolder, variant, outputSize, modelType, executionProvider), logger);
        }


        /// <summary>
        /// Create Nitro pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.[512, 512-Turbo, 1024, 1024-Turbo]</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        public static NitroPipeline FromFolder(string modelFolder, string variant, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new NitroPipeline(NitroConfig.FromFolder(modelFolder, variant, executionProvider), logger);
        }
    }
}
