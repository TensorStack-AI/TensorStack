// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.IO;
using System.Linq;
using TensorStack.Common;
using TensorStack.StableDiffusion.Config;
using TensorStack.StableDiffusion.Enums;
using TensorStack.TextGeneration.Tokenizers;

namespace TensorStack.StableDiffusion.Pipelines.Flux
{
    public record FluxConfig : PipelineConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="FluxConfig"/> class.
        /// </summary>
        public FluxConfig()
        {
            Tokenizer = new TokenizerConfig();
            Tokenizer2 = new TokenizerConfig{MaxLength = 512 };
            TextEncoder = new CLIPModelConfig();
            TextEncoder2 = new CLIPModelConfig
            {
                PadTokenId = 0,
                HiddenSize = 4096,
                IsFixedSequenceLength = false,
                SequenceLength = 512
            };
            Transformer = new TransformerModelConfig
            {
                JointAttention = 4096,
                PooledProjection = 768,
                IsOptimizationSupported = true
            };
            AutoEncoder = new AutoEncoderModelConfig
            {
                LatentChannels = 16,
                ScaleFactor = 0.3611f,
                ShiftFactor = 0.1159f
            };
        }

        public string Name { get; init; } = "Flux";
        public override PipelineType Pipeline { get; } = PipelineType.Flux;
        public TokenizerConfig Tokenizer { get; init; }
        public TokenizerConfig Tokenizer2 { get; init; }
        public CLIPModelConfig TextEncoder { get; init; }
        public CLIPModelConfig TextEncoder2 { get; init; }
        public TransformerModelConfig Transformer { get; init; }
        public AutoEncoderModelConfig AutoEncoder { get; init; }


        /// <summary>
        /// Sets the execution provider for all models.
        /// </summary>
        /// <param name="executionProvider">The execution provider.</param>
        public override void SetProvider(ExecutionProvider executionProvider)
        {
            TextEncoder.SetProvider(executionProvider);
            TextEncoder2.SetProvider(executionProvider);
            Transformer.SetProvider(executionProvider);
            AutoEncoder.SetProvider(executionProvider);
        }


        /// <summary>
        /// Saves the configuration to file.
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="useRelativePaths">if set to <c>true</c> use relative paths.</param>
        public override void Save(string configFile, bool useRelativePaths = true)
        {
            ConfigService.Serialize(configFile, this, useRelativePaths);
        }


        /// <summary>
        /// Create Flux configuration from default values
        /// </summary>
        /// <param name="name">The name.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>FluxConfig.</returns>
        public static FluxConfig FromDefault(string name, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            var config = new FluxConfig { Name = name };
            config.Transformer.ModelType = modelType;
            config.SetProvider(executionProvider);
            return config;
        }


        /// <summary>
        /// Create StableDiffusionv configuration from json file
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>FluxConfig.</returns>
        public static FluxConfig FromFile(string configFile, ExecutionProvider executionProvider = default)
        {
            var config = ConfigService.Deserialize<FluxConfig>(configFile);
            config.SetProvider(executionProvider);
            return config;
        }


        /// <summary>
        /// Create Flux configuration from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>FluxConfig.</returns>
        public static FluxConfig FromFolder(string modelFolder, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            return CreateFromFolder(modelFolder, default, modelType, executionProvider);
        }


        /// <summary>
        /// Create Flux configuration from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>FluxConfig.</returns>
        public static FluxConfig FromFolder(string modelFolder, string variant, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            return CreateFromFolder(modelFolder, variant, modelType, executionProvider);
        }


        /// <summary>
        /// Create Flux configuration from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>FluxConfig.</returns>
        public static FluxConfig FromFolder(string modelFolder, string variant, ExecutionProvider executionProvider = default)
        {
            string[] typeOptions = ["Turbo", "Distilled", "Dist", "Schnell"];
            var modelType = typeOptions.Any(v => variant.Contains(v, StringComparison.OrdinalIgnoreCase)) ? ModelType.Turbo : ModelType.Base;
            return CreateFromFolder(modelFolder, variant, modelType, executionProvider);
        }


        /// <summary>
        /// Create Flux configuration from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>FluxConfig.</returns>
        private static FluxConfig CreateFromFolder(string modelFolder, string variant, ModelType modelType, ExecutionProvider executionProvider)
        {
            var config = FromDefault(Path.GetFileNameWithoutExtension(modelFolder), modelType, executionProvider);
            config.Tokenizer.Path = Path.Combine(modelFolder, "tokenizer", "vocab.json");
            config.Tokenizer2.Path = Path.Combine(modelFolder, "tokenizer_2", "spiece.model");
            config.TextEncoder.Path = GetVariantPath(modelFolder, "text_encoder", "model.onnx", variant);
            config.TextEncoder2.Path = GetVariantPath(modelFolder, "text_encoder_2", "model.onnx", variant);
            config.Transformer.Path = GetVariantPath(modelFolder, "transformer", "model.onnx", variant);
            config.AutoEncoder.DecoderModelPath = GetVariantPath(modelFolder, "vae_decoder", "model.onnx", variant);
            config.AutoEncoder.EncoderModelPath = GetVariantPath(modelFolder, "vae_encoder", "model.onnx", variant);
            var controlNetPath = GetVariantPath(modelFolder, "transformer", "controlnet.onnx", variant);
            if (File.Exists(controlNetPath))
                config.Transformer.ControlNetPath = controlNetPath;
            return config;
        }

    }
}
