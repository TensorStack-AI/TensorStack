// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.IO;
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
                Scale = 32
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
        public static NitroConfig FromDefault(string name, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            var config = new NitroConfig { Name = name };
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
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>NitroConfig.</returns>
        public static NitroConfig FromFolder(string modelFolder, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            var config = FromDefault(Path.GetFileNameWithoutExtension(modelFolder), modelType, executionProvider);
            config.Tokenizer.Path = Path.Combine(modelFolder, "tokenizer");
            config.TextEncoder.Path = Path.Combine(modelFolder, "text_encoder", "model.onnx");
            config.Transformer.Path = Path.Combine(modelFolder, "transformer", "model.onnx");
            config.AutoEncoder.DecoderModelPath = Path.Combine(modelFolder, "vae_decoder", "model.onnx");
            config.AutoEncoder.EncoderModelPath = Path.Combine(modelFolder, "vae_encoder", "model.onnx");
            return config;
        }

    }
}
