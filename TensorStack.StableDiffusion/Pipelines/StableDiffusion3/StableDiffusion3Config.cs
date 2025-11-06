// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.IO;
using TensorStack.Common;
using TensorStack.StableDiffusion.Config;
using TensorStack.StableDiffusion.Enums;
using TensorStack.TextGeneration.Tokenizers;

namespace TensorStack.StableDiffusion.Pipelines.StableDiffusion3
{
    public record StableDiffusion3Config : PipelineConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusion3Config"/> class.
        /// </summary>
        public StableDiffusion3Config()
        {
            Tokenizer = new TokenizerConfig();
            Tokenizer2 = new TokenizerConfig();
            Tokenizer3 = new TokenizerConfig();
            TextEncoder = new CLIPModelConfig();
            TextEncoder2 = new CLIPModelConfig { HiddenSize = 1280 };
            TextEncoder3 = new CLIPModelConfig
            {
                PadTokenId = 0,
                HiddenSize = 4096,
                IsFixedSequenceLength = false,
                SequenceLength = 512
            };
            Transformer = new TransformerModelConfig
            {
                JointAttention = 4096,
                PooledProjection = 2048,
                CaptionProjection = 1536,
                IsOptimizationSupported = true
            };
            AutoEncoder = new AutoEncoderModelConfig
            {
                LatentChannels = 16,
                ScaleFactor = 1.5305f,
                ShiftFactor = 0.0609f
            };
        }

        public string Name { get; init; } = "StableDiffusion3";
        public override PipelineType Pipeline { get; } = PipelineType.StableDiffusion3;
        public TokenizerConfig Tokenizer { get; init; }
        public TokenizerConfig Tokenizer2 { get; init; }
        public TokenizerConfig Tokenizer3 { get; init; }
        public CLIPModelConfig TextEncoder { get; init; }
        public CLIPModelConfig TextEncoder2 { get; init; }
        public CLIPModelConfig TextEncoder3 { get; init; }
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
            TextEncoder3.SetProvider(executionProvider);
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
        /// Create StableDiffusion3 configuration from default values
        /// </summary>
        /// <param name="name">The name.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>StableDiffusion3Config.</returns>
        public static StableDiffusion3Config FromDefault(string name, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            var config = new StableDiffusion3Config { Name = name };
            config.Transformer.ModelType = modelType;
            config.SetProvider(executionProvider);
            return config;
        }


        /// <summary>
        /// Create StableDiffusionv configuration from json file
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>StableDiffusion3Config.</returns>
        public static StableDiffusion3Config FromFile(string configFile, ExecutionProvider executionProvider = default)
        {
            var config = ConfigService.Deserialize<StableDiffusion3Config>(configFile);
            config.SetProvider(executionProvider);
            return config;
        }


        /// <summary>
        /// Create StableDiffusion3 configuration from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>StableDiffusion3Config.</returns>
        public static StableDiffusion3Config FromFolder(string modelFolder, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            var config = FromDefault(Path.GetFileNameWithoutExtension(modelFolder), modelType, executionProvider);
            config.Tokenizer.Path = Path.Combine(modelFolder, "tokenizer", "vocab.json");
            config.Tokenizer2.Path = Path.Combine(modelFolder, "tokenizer_2", "vocab.json");
            config.Tokenizer3.Path = Path.Combine(modelFolder, "tokenizer_3", "spiece.model");
            config.TextEncoder.Path = Path.Combine(modelFolder, "text_encoder", "model.onnx");
            config.TextEncoder2.Path = Path.Combine(modelFolder, "text_encoder_2", "model.onnx");
            config.TextEncoder3.Path = Path.Combine(modelFolder, "text_encoder_3", "model.onnx");
            config.Transformer.Path = Path.Combine(modelFolder, "transformer", "model.onnx");
            config.AutoEncoder.DecoderModelPath = Path.Combine(modelFolder, "vae_decoder", "model.onnx");
            config.AutoEncoder.EncoderModelPath = Path.Combine(modelFolder, "vae_encoder", "model.onnx");
            var controlNetPath = Path.Combine(modelFolder, "controlnet", "model.onnx");
            if (File.Exists(controlNetPath))
                config.Transformer.ControlNetPath = controlNetPath;
            return config;
        }

    }
}
