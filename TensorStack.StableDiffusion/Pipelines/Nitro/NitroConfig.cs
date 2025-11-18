// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.IO;
using System.Linq;
using TensorStack.Common;
using TensorStack.StableDiffusion.Config;
using TensorStack.StableDiffusion.Enums;
using TensorStack.TextGeneration.Common;
using TensorStack.TextGeneration.Tokenizers;

namespace TensorStack.StableDiffusion.Pipelines.Nitro
{
    public record NitroConfig : PipelineConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="NitroConfig"/> class.
        /// </summary>
        public NitroConfig()
        {
            OutputSize = 512;
            Tokenizer = new TokenizerConfig
            {
                BOS = 128000,
                EOS = 128001
            };
            TextEncoder = new DecoderConfig
            {
                NumHeads = 32,
                NumLayers = 16,
                NumKVHeads = 8,
                HiddenSize = 2048,
                VocabSize = 128256
            };
            Transformer = new TransformerModelConfig
            {
                InChannels = 32,
                OutChannels = 32,
                JointAttention = 2048,
                IsOptimizationSupported = true
            };
            AutoEncoder = new AutoEncoderModelConfig
            {
                Scale = 32,
                LatentChannels = 32,
                ScaleFactor = 0.41407f
            };
        }

        public string Name { get; init; } = "Nitro";
        public override PipelineType Pipeline { get; } = PipelineType.Nitro;
        public TokenizerConfig Tokenizer { get; init; }
        public DecoderConfig TextEncoder { get; init; }
        public TransformerModelConfig Transformer { get; init; }
        public AutoEncoderModelConfig AutoEncoder { get; init; }
        public int OutputSize { get; init; }


        /// <summary>
        /// Sets the execution provider for all models.
        /// </summary>
        /// <param name="executionProvider">The execution provider.</param>
        public override void SetProvider(ExecutionProvider executionProvider)
        {
            TextEncoder.SetProvider(executionProvider);
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
        /// Create Nitro configuration from default values
        /// </summary>
        /// <param name="name">The name.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>NitroConfig.</returns>
        public static NitroConfig FromDefault(string name, int outputSize, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            var config = new NitroConfig { Name = name, OutputSize = outputSize };
            config.Transformer.ModelType = modelType;
            config.SetProvider(executionProvider);
            return config;
        }


        /// <summary>
        /// Create StableDiffusionv configuration from json file
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>NitroConfig.</returns>
        public static NitroConfig FromFile(string configFile, ExecutionProvider executionProvider = default)
        {
            var config = ConfigService.Deserialize<NitroConfig>(configFile);
            config.SetProvider(executionProvider);
            return config;
        }


        /// <summary>
        /// Create Nitro configuration from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="outputSize">Size of the output.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        public static NitroConfig FromFolder(string modelFolder, int outputSize, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            return CreateFromFolder(modelFolder, default, outputSize, modelType, executionProvider);
        }


        /// <summary>
        /// Create Nitro configuration from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="outputSize">Size of the output.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>NitroConfig.</returns>
        public static NitroConfig FromFolder(string modelFolder, string variant, int outputSize, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            return CreateFromFolder(modelFolder, variant, outputSize, modelType, executionProvider);
        }


        /// <summary>
        /// Create Nitro configuration from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>NitroConfig.</returns>
        public static NitroConfig FromFolder(string modelFolder, string variant, ExecutionProvider executionProvider = default)
        {
            string[] sizeOptions = ["XL", "Large", "1024"];
            string[] typeOptions = ["Turbo", "Distilled", "Dist"];
            var outputSize = sizeOptions.Any(v => variant.Contains(v, StringComparison.OrdinalIgnoreCase)) ? 1024 : 512;
            var modelType = typeOptions.Any(v => variant.Contains(v, StringComparison.OrdinalIgnoreCase)) ? ModelType.Turbo : ModelType.Base;
            return CreateFromFolder(modelFolder, variant, outputSize, modelType, executionProvider);
        }


        /// <summary>
        /// Create Nitro configuration from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="outputSize">Size of the output.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>NitroConfig.</returns>
        private static NitroConfig CreateFromFolder(string modelFolder, string variant, int outputSize, ModelType modelType, ExecutionProvider executionProvider)
        {
            var config = FromDefault(Path.GetFileNameWithoutExtension(modelFolder), outputSize, modelType, executionProvider);
            config.Tokenizer.Path = Path.Combine(modelFolder, "tokenizer");
            config.TextEncoder.Path = GetVariantPath(modelFolder, "text_encoder", "model.onnx", variant);
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
