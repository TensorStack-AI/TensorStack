// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.IO;
using TensorStack.Common;
using TensorStack.StableDiffusion.Config;
using TensorStack.StableDiffusion.Enums;

namespace TensorStack.StableDiffusion.Pipelines.StableCascade
{
    public record StableCascadeConfig : PipelineConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="StableCascadeConfig"/> class.
        /// </summary>
        public StableCascadeConfig()
        {
            Tokenizer = new TokenizerConfig();
            PriorUnet = new UNetModelConfig();
            DecoderUnet = new UNetModelConfig();
            TextEncoder = new CLIPModelConfig { HiddenSize = 1280 };
            ImageEncoder = new CLIPModelConfig { HiddenSize = 768 };
            ImageDecoder = new PaellaVQModelConfig
            {
                Scale = 4,
                ScaleFactor = 0.3764f
            };
        }

        public string Name { get; init; } = "StableCascade";
        public override PipelineType Pipeline { get; } = PipelineType.StableCascade;
        public TokenizerConfig Tokenizer { get; init; }
        public CLIPModelConfig TextEncoder { get; init; }
        public UNetModelConfig PriorUnet { get; init; }
        public UNetModelConfig DecoderUnet { get; init; }
        public PaellaVQModelConfig ImageDecoder { get; init; }
        public CLIPModelConfig ImageEncoder { get; init; }


        /// <summary>
        /// Sets the execution provider for all models.
        /// </summary>
        /// <param name="executionProvider">The execution provider.</param>
        public override void SetProvider(ExecutionProvider executionProvider)
        {
            Tokenizer.SetProvider(executionProvider);
            TextEncoder.SetProvider(executionProvider);
            PriorUnet.SetProvider(executionProvider);
            DecoderUnet.SetProvider(executionProvider);
            ImageEncoder.SetProvider(executionProvider);
            ImageDecoder.SetProvider(executionProvider);
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
        /// Create StableCascade configuration from default values
        /// </summary>
        /// <param name="name">The name.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>StableCascadeConfig.</returns>
        public static StableCascadeConfig FromDefault(string name, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            var config = new StableCascadeConfig { Name = name };
            config.PriorUnet.ModelType = modelType;
            config.DecoderUnet.ModelType = modelType;
            config.SetProvider(executionProvider);
            return config;
        }


        /// <summary>
        /// Create StableCascade configuration from json file
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>StableCascadeConfig.</returns>
        public static StableCascadeConfig FromFile(string configFile, ExecutionProvider executionProvider = default)
        {
            var config = ConfigService.Deserialize<StableCascadeConfig>(configFile);
            config.SetProvider(executionProvider);
            return config;
        }


        /// <summary>
        /// Create StableCascade configuration from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>StableCascadeConfig.</returns>
        public static StableCascadeConfig FromFolder(string modelFolder, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            var config = FromDefault(Path.GetFileNameWithoutExtension(modelFolder), modelType, executionProvider);
            config.Tokenizer.Path = Path.Combine(modelFolder, "tokenizer", "vocab.json");
            config.TextEncoder.Path = Path.Combine(modelFolder, "text_encoder", "model.onnx");
            config.PriorUnet.Path = Path.Combine(modelFolder, "prior", "model.onnx");
            config.DecoderUnet.Path = Path.Combine(modelFolder, "decoder", "model.onnx");
            config.ImageEncoder.Path = Path.Combine(modelFolder, "vae_encoder", "model.onnx");
            config.ImageDecoder.Path = Path.Combine(modelFolder, "vae_decoder", "model.onnx");
            return config;
        }

    }
}
