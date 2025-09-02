// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.Extensions.Logging;
using System.Collections.Generic;
using TensorStack.Common;
using TensorStack.StableDiffusion.Common;
using TensorStack.StableDiffusion.Enums;
using TensorStack.StableDiffusion.Models;
using TensorStack.StableDiffusion.Pipelines.StableDiffusion;
using TensorStack.StableDiffusion.Tokenizers;

namespace TensorStack.StableDiffusion.Pipelines.LatentConsistency
{
    public class LatentConsistencyPipeline : StableDiffusionPipeline
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LatentConsistencyPipeline"/> class.
        /// </summary>
        /// <param name="unet">The unet.</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="autoEncoder">The automatic encoder.</param>
        /// <param name="logger">The logger.</param>
        public LatentConsistencyPipeline(UNetConditionalModel unet, CLIPTokenizer tokenizer, CLIPTextModel textEncoder, AutoEncoderModel autoEncoder, ILogger logger = null)
            : base(unet, tokenizer, textEncoder, autoEncoder, logger) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="LatentConsistencyPipeline"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="logger">The logger.</param>
        public LatentConsistencyPipeline(LatentConsistencyConfig configuration, ILogger logger = null)
            : base(configuration, logger) { }


        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override PipelineType PipelineType => PipelineType.LatentConsistency;

        /// <summary>
        /// Gets the friendly name.
        /// </summary>
        public override string Name { get; init; } = nameof(PipelineType.LatentConsistency);


        /// <summary>
        /// Configures the supported schedulers.
        /// </summary>
        protected override IReadOnlyList<SchedulerType> ConfigureSchedulers()
        {
            return [SchedulerType.LCM];
        }


        /// <summary>
        /// Configures the default SchedulerOptions.
        /// </summary>
        protected override GenerateOptions ConfigureDefaultOptions()
        {
            return new GenerateOptions
            {
                Steps = 4,
                Width = 512,
                Height = 512,
                GuidanceScale = 0f,
                Scheduler = SchedulerType.LCM
            };
        }


        /// <summary>
        /// Create LatentConsistency pipeline from LatentConsistencyConfig file
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>LatentConsistencyPipeline.</returns>
        public static new LatentConsistencyPipeline FromConfig(string configFile, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new LatentConsistencyPipeline(LatentConsistencyConfig.FromFile(configFile, executionProvider), logger);
        }


        /// <summary>
        /// Create LatentConsistency pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>LatentConsistencyPipeline.</returns>
        public static new LatentConsistencyPipeline FromFolder(string modelFolder, ModelType modelType, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new LatentConsistencyPipeline(LatentConsistencyConfig.FromFolder(modelFolder, modelType, executionProvider), logger);
        }
    }
}
