// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.IO;
using TensorStack.Common;
using TensorStack.StableDiffusion.Config;
using TensorStack.StableDiffusion.Enums;

namespace TensorStack.StableDiffusion.Pipelines.StableDiffusionXL
{
    public record StableDiffusionXLConfig : PipelineConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="StableDiffusionXLConfig"/> class.
        /// </summary>
        public StableDiffusionXLConfig()
        {
            Tokenizer = new TokenizerConfig();
            Tokenizer2 = new TokenizerConfig();
            TextEncoder = new CLIPModelConfig { HiddenSize = 768 };
            TextEncoder2 = new CLIPModelConfig { HiddenSize = 1280 };
            Unet = new UNetModelConfig { IsOptimizationSupported = true };
            AutoEncoder = new AutoEncoderModelConfig { ScaleFactor = 0.13025f };
        }

        public string Name { get; init; } = "StableDiffusionXL";
        public override PipelineType Pipeline { get; } = PipelineType.StableDiffusionXL;
        public TokenizerConfig Tokenizer { get; init; }
        public TokenizerConfig Tokenizer2 { get; init; }
        public CLIPModelConfig TextEncoder { get; init; }
        public CLIPModelConfig TextEncoder2 { get; init; }
        public UNetModelConfig Unet { get; init; }
        public AutoEncoderModelConfig AutoEncoder { get; init; }


        /// <summary>
        /// Sets the execution provider for all models.
        /// </summary>
        /// <param name="executionProvider">The execution provider.</param>
        public override void SetProvider(ExecutionProvider executionProvider)
        {
            Tokenizer.SetProvider(executionProvider);
            Tokenizer2.SetProvider(executionProvider);
            TextEncoder.SetProvider(executionProvider);
            TextEncoder2.SetProvider(executionProvider);
            Unet.SetProvider(executionProvider);
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
        /// Create StableDiffusion configuration from default values
        /// </summary>
        /// <param name="name">The name.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>StableDiffusionXLConfig.</returns>
        public static StableDiffusionXLConfig FromDefault(string name, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            var config = new StableDiffusionXLConfig { Name = name };
            config.Unet.ModelType = modelType;
            config.SetProvider(executionProvider);
            return config;
        }


        /// <summary>
        /// Create StableDiffusion configuration from json file
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>StableDiffusionXLConfig.</returns>
        public static StableDiffusionXLConfig FromFile(string configFile, ExecutionProvider executionProvider = default)
        {
            var config = ConfigService.Deserialize<StableDiffusionXLConfig>(configFile);
            config.SetProvider(executionProvider);
            return config;
        }


        /// <summary>
        /// Create StableDiffusion configuration from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>StableDiffusionXLConfig.</returns>
        public static StableDiffusionXLConfig FromFolder(string modelFolder, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            var config = FromDefault(Path.GetFileNameWithoutExtension(modelFolder), modelType, executionProvider);
            config.Tokenizer.Path = Path.Combine(modelFolder, "tokenizer", "vocab.json");
            config.Tokenizer2.Path = Path.Combine(modelFolder, "tokenizer_2", "vocab.json");
            config.TextEncoder.Path = Path.Combine(modelFolder, "text_encoder", "model.onnx");
            config.TextEncoder2.Path = Path.Combine(modelFolder, "text_encoder_2", "model.onnx");
            config.Unet.Path = Path.Combine(modelFolder, "unet", "model.onnx");
            config.AutoEncoder.DecoderModelPath = Path.Combine(modelFolder, "vae_decoder", "model.onnx");
            config.AutoEncoder.EncoderModelPath = Path.Combine(modelFolder, "vae_encoder", "model.onnx");
            var controlNetPath = Path.Combine(modelFolder, "unet", "controlnet.onnx");
            if (File.Exists(controlNetPath))
                config.Unet.ControlNetPath = controlNetPath;
            return config;
        }

    }
}
