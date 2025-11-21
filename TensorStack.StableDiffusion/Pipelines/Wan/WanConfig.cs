// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.IO;
using System.Linq;
using TensorStack.Common;
using TensorStack.StableDiffusion.Config;
using TensorStack.StableDiffusion.Enums;
using TensorStack.TextGeneration.Tokenizers;

namespace TensorStack.StableDiffusion.Pipelines.Wan
{
    public record WanConfig : PipelineConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="WanConfig"/> class.
        /// </summary>
        public WanConfig()
        {
            Tokenizer = new TokenizerConfig();
            TextEncoder = new CLIPModelConfig
            {
                PadTokenId = 0,
                HiddenSize = 4096,
                SequenceLength = 512,
                IsFixedSequenceLength = true,
            };
            Transformer = new TransformerModelConfig
            {
                JointAttention = 4096,
                IsOptimizationSupported = true
            };
            AutoEncoder = new AutoEncoderModelConfig
            {
                LatentChannels = 16,
                ScaleFactor = 1,
                ShiftFactor = 0,
                LatentsMean =
                [
                    -0.7571f,
                    -0.7089f,
                    -0.9113f,
                    0.1075f,
                    -0.1745f,
                    0.9653f,
                    -0.1517f,
                    1.5508f,
                    0.4134f,
                    -0.0715f,
                    0.5517f,
                    -0.3632f,
                    -0.1922f,
                    -0.9497f,
                    0.2503f,
                    -0.2921f
                ],
                LatentsStd =
                [
                    2.8184f,
                    1.4541f,
                    2.3275f,
                    2.6558f,
                    1.2196f,
                    1.7708f,
                    2.6052f,
                    2.0743f,
                    3.2687f,
                    2.1526f,
                    2.8652f,
                    1.5579f,
                    1.6382f,
                    1.1253f,
                    2.8251f,
                    1.916f
                ]
            };
        }

        public string Name { get; init; } = "Wan";
        public override PipelineType Pipeline { get; } = PipelineType.Wan;
        public TokenizerConfig Tokenizer { get; init; }
        public CLIPModelConfig TextEncoder { get; init; }
        public TransformerModelConfig Transformer { get; init; }
        public AutoEncoderModelConfig AutoEncoder { get; init; }


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
        /// Create Wan configuration from default values
        /// </summary>
        /// <param name="name">The name.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>WanConfig.</returns>
        public static WanConfig FromDefault(string name, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            var config = new WanConfig { Name = name };
            config.Transformer.ModelType = modelType;
            config.SetProvider(executionProvider);
            return config;
        }


        /// <summary>
        /// Create StableDiffusionv configuration from json file
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>WanConfig.</returns>
        public static WanConfig FromFile(string configFile, ExecutionProvider executionProvider = default)
        {
            var config = ConfigService.Deserialize<WanConfig>(configFile);
            config.SetProvider(executionProvider);
            return config;
        }


        /// <summary>
        /// Create Wan configuration from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>WanConfig.</returns>
        public static WanConfig FromFolder(string modelFolder, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            return CreateFromFolder(modelFolder, default, modelType, executionProvider);
        }


        /// <summary>
        /// Create Wan configuration from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>WanConfig.</returns>
        public static WanConfig FromFolder(string modelFolder, string variant, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            return CreateFromFolder(modelFolder, variant, modelType, executionProvider);
        }


        /// <summary>
        /// Create Wan configuration from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>WanConfig.</returns>
        public static WanConfig FromFolder(string modelFolder, string variant, ExecutionProvider executionProvider = default)
        {
            string[] typeOptions = ["Turbo", "Distilled", "Dist", "Flash"];
            var modelType = typeOptions.Any(v => variant.Contains(v, StringComparison.OrdinalIgnoreCase)) ? ModelType.Turbo : ModelType.Base;
            return CreateFromFolder(modelFolder, variant, modelType, executionProvider);
        }


        /// <summary>
        /// Create Wan configuration from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>WanConfig.</returns>
        private static WanConfig CreateFromFolder(string modelFolder, string variant, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            var config = FromDefault(Path.GetFileNameWithoutExtension(modelFolder), modelType, executionProvider);
            config.Tokenizer.Path = Path.Combine(modelFolder, "tokenizer", "spiece.model");
            config.TextEncoder.Path = GetVariantPath(modelFolder, "text_encoder", "model.onnx", variant);
            config.Transformer.Path = GetVariantPath(modelFolder, "transformer", "model.onnx", variant);
            config.AutoEncoder.DecoderModelPath = GetVariantPath(modelFolder, "vae_decoder", "model.onnx", variant);
            //config.AutoEncoder.EncoderModelPath = GetVariantPath(modelFolder, "vae_encoder", "model.onnx", variant);
            return config;
        }
    }
}
