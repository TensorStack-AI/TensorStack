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

namespace TensorStack.StableDiffusion.Pipelines.StableDiffusion2
{
    public class StableDiffusion2Pipeline : StableDiffusionPipeline
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusion2Pipeline"/> class.
        /// </summary>
        /// <param name="unet">The unet.</param>
        /// <param name="tokenizer">The tokenizer.</param>
        /// <param name="textEncoder">The text encoder.</param>
        /// <param name="autoEncoder">The automatic encoder.</param>
        /// <param name="logger">The logger.</param>
        public StableDiffusion2Pipeline(UNetConditionalModel unet, CLIPTokenizer tokenizer, CLIPTextModel textEncoder, AutoEncoderModel autoEncoder, ILogger logger = default)
            : base(unet, tokenizer, textEncoder, autoEncoder, logger)
        {
            Unet = unet;
            Tokenizer = tokenizer;
            TextEncoder = textEncoder;
            AutoEncoder = autoEncoder;
            Initialize();
            Logger?.LogInformation("[StableDiffusion2Pipeline] Name: {Name}", Name);
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusion2Pipeline"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="logger">The logger.</param>
        public StableDiffusion2Pipeline(StableDiffusion2Config configuration, ILogger logger = default) : this(
            new UNetConditionalModel(configuration.Unet),
            new CLIPTokenizer(configuration.Tokenizer),
            new CLIPTextModel(configuration.TextEncoder),
            new AutoEncoderModel(configuration.AutoEncoder),
            logger)
        {
            Name = configuration.Name;
        }

        /// <summary>
        /// Gets the type of the pipeline.
        /// </summary>
        public override PipelineType PipelineType => PipelineType.StableDiffusion2;

        /// <summary>
        /// Gets the friendly name.
        /// </summary>
        public override string Name { get; init; } = nameof(PipelineType.StableDiffusion2);


        /// <summary>
        /// Configures the supported schedulers.
        /// </summary>
        protected override IReadOnlyList<SchedulerType> ConfigureSchedulers()
        {
            return [SchedulerType.LMS, SchedulerType.Euler, SchedulerType.EulerAncestral, SchedulerType.LCM];
        }


        /// <summary>
        /// Configures the default SchedulerOptions.
        /// </summary>
        protected override GenerateOptions ConfigureDefaultOptions()
        {
            return new GenerateOptions
            {
                Steps = 30,
                Width = 768,
                Height = 768,
                GuidanceScale = 7.5f,
                Scheduler = SchedulerType.Euler,
                PredictionType = PredictionType.VariablePrediction
            };
        }


        /// <summary>
        /// Create StableDiffusion2 pipeline from StableDiffusion2Config file
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>StableDiffusion2Pipeline.</returns>
        public static new StableDiffusion2Pipeline FromConfig(string configFile, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new StableDiffusion2Pipeline(StableDiffusion2Config.FromFile(configFile, executionProvider), logger);
        }


        /// <summary>
        /// Create StableDiffusion2 pipeline from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="logger">The logger.</param>
        /// <returns>StableDiffusion2Pipeline.</returns>
        public static new StableDiffusion2Pipeline FromFolder(string modelFolder, ModelType modelType, ExecutionProvider executionProvider, ILogger logger = default)
        {
            return new StableDiffusion2Pipeline(StableDiffusion2Config.FromFolder(modelFolder, modelType, executionProvider), logger);
        }
    }
}
