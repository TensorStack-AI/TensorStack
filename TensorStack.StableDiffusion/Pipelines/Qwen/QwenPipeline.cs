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
using QwenTextPipeline = TensorStack.TextGeneration.Pipelines.Qwen.QwenPipeline;

namespace TensorStack.StableDiffusion.Pipelines.Qwen
{
    public class QwenPipeline : QwenBase, IPipeline<ImageTensor, GenerateOptions, GenerateProgress>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="QwenPipeline"/> class.
        /// </summary>
        /// <param name="transformer">The transformer.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="autoEncoder">The automatic encoder.</param>
        /// <param name="logger">The logger.</param>
        public QwenPipeline(TransformerQwenModel transformer, QwenTextPipeline textEncoder, AutoEncoderModel autoEncoder, ILogger logger = null)
            : base(transformer, textEncoder, autoEncoder, logger) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="QwenPipeline"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="logger">The logger.</param>
        public QwenPipeline(QwenConfig configuration, ILogger logger = null)
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
        /// Create Qwen pipeline from StableDiffusionConfig file
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>QwenPipeline.</returns>
        public static QwenPipeline FromConfig(string configFile, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new QwenPipeline(QwenConfig.FromFile(configFile, executionProvider), logger);
        }


        /// <summary>
        /// Create Qwen pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>QwenPipeline.</returns>
        public static QwenPipeline FromFolder(string modelFolder, ModelType modelType, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new QwenPipeline(QwenConfig.FromFolder(modelFolder, modelType, executionProvider), logger);
        }


        /// <summary>
        /// Create Qwen pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>QwenPipeline.</returns>
        public static QwenPipeline FromFolder(string modelFolder, string variant, ModelType modelType, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new QwenPipeline(QwenConfig.FromFolder(modelFolder, variant, modelType, executionProvider), logger);
        }


        /// <summary>
        /// Create Qwen pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>QwenPipeline.</returns>
        public static QwenPipeline FromFolder(string modelFolder, string variant, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new QwenPipeline(QwenConfig.FromFolder(modelFolder, variant, executionProvider), logger);
        }
    }
}
