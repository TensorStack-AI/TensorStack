// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.Extensions.Logging;
using System.Collections.Generic;
using TensorStack.Common;
using TensorStack.StableDiffusion.Common;
using TensorStack.StableDiffusion.Enums;
using TensorStack.StableDiffusion.Models;
using TensorStack.StableDiffusion.Pipelines.StableDiffusion;
using TensorStack.TextGeneration.Tokenizers;

namespace TensorStack.StableDiffusion.Pipelines.LatentConsistency
{
    public class LatentConsistencyVideoPipeline : StableDiffusionVideoPipeline
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LatentConsistencyVideoPipeline"/> class.
        /// </summary>
        /// <param name="unet">The unet.</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="autoEncoder">The automatic encoder.</param>
        /// <param name="logger">The logger.</param>
        public LatentConsistencyVideoPipeline(UNetConditionalModel unet, CLIPTokenizer tokenizer, CLIPTextModel textEncoder, AutoEncoderModel autoEncoder, ILogger logger = null)
            : base(unet, tokenizer, textEncoder, autoEncoder, logger) { }

        /// <summary>
        /// Initializes a new instance of the <see cref="LatentConsistencyVideoPipeline"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="logger">The logger.</param>
        public LatentConsistencyVideoPipeline(LatentConsistencyConfig configuration, ILogger logger = null)
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
        /// Create LatentConsistencyVideo pipeline from LatentConsistencyConfig file
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>LatentConsistencyVideoPipeline.</returns>
        public static new LatentConsistencyVideoPipeline FromConfig(string configFile, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new LatentConsistencyVideoPipeline(LatentConsistencyConfig.FromFile(configFile, executionProvider), logger);
        }


        /// <summary>
        /// Create LatentConsistencyVideoPipeline pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>LatentConsistencyVideoPipeline.</returns>
        public static new LatentConsistencyVideoPipeline FromFolder(string modelFolder, ModelType modelType, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new LatentConsistencyVideoPipeline(LatentConsistencyConfig.FromFolder(modelFolder, modelType, executionProvider), logger);
        }


        /// <summary>
        /// Create LatentConsistencyVideoPipeline pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>LatentConsistencyVideoPipeline.</returns>
        public static new LatentConsistencyVideoPipeline FromFolder(string modelFolder, string variant, ModelType modelType, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new LatentConsistencyVideoPipeline(LatentConsistencyConfig.FromFolder(modelFolder, variant, modelType, executionProvider), logger);
        }

    }
}
