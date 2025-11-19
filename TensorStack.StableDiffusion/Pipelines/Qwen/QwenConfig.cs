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

namespace TensorStack.StableDiffusion.Pipelines.Qwen
{
    public record QwenConfig : PipelineConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="QwenConfig"/> class.
        /// </summary>
        public QwenConfig()
        {
            Tokenizer = new TokenizerConfig
            {
                BOS = 151643,
                EOS = 151645
            };
            TextEncoder = new DecoderConfig
            {
                NumHeads = 28,
                NumLayers = 28,
                NumKVHeads = 4,
                HiddenSize = 3584,
                VocabSize = 152064
            };
            Transformer = new TransformerModelConfig
            {
                InChannels = 64,
                OutChannels = 16,
                JointAttention = 3584,
                PooledProjection = 768,
                IsOptimizationSupported = true
            };
            AutoEncoder = new AutoEncoderModelConfig
            {
                Scale = 16,
                LatentChannels = 16,
                ScaleFactor = 1,
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

        public string Name { get; init; } = "Qwen";
        public override PipelineType Pipeline { get; } = PipelineType.Qwen;
        public TokenizerConfig Tokenizer { get; init; }
        public DecoderConfig TextEncoder { get; init; }
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
        /// Create Qwen configuration from default values
        /// </summary>
        /// <param name="name">The name.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>QwenConfig.</returns>
        public static QwenConfig FromDefault(string name, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            var config = new QwenConfig { Name = name };
            config.Transformer.ModelType = modelType;
            config.SetProvider(executionProvider);
            return config;
        }


        /// <summary>
        /// Create StableDiffusionv configuration from json file
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>QwenConfig.</returns>
        public static QwenConfig FromFile(string configFile, ExecutionProvider executionProvider = default)
        {
            var config = ConfigService.Deserialize<QwenConfig>(configFile);
            config.SetProvider(executionProvider);
            return config;
        }


        /// <summary>
        /// Create Qwen configuration from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        public static QwenConfig FromFolder(string modelFolder, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            return CreateFromFolder(modelFolder, default, modelType, executionProvider);
        }


        /// <summary>
        /// Create Qwen configuration from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>QwenConfig.</returns>
        public static QwenConfig FromFolder(string modelFolder, string variant, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            return CreateFromFolder(modelFolder, variant, modelType, executionProvider);
        }


        /// <summary>
        /// Create Qwen configuration from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>QwenConfig.</returns>
        public static QwenConfig FromFolder(string modelFolder, string variant, ExecutionProvider executionProvider = default)
        {
            string[] typeOptions = ["Turbo", "Distilled", "Dist"];
            var modelType = typeOptions.Any(v => variant.Contains(v, StringComparison.OrdinalIgnoreCase)) ? ModelType.Turbo : ModelType.Base;
            return CreateFromFolder(modelFolder, variant, modelType, executionProvider);
        }


        /// <summary>
        /// Create Qwen configuration from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>QwenConfig.</returns>
        private static QwenConfig CreateFromFolder(string modelFolder, string variant, ModelType modelType, ExecutionProvider executionProvider)
        {
            var config = FromDefault(Path.GetFileNameWithoutExtension(modelFolder), modelType, executionProvider);
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
