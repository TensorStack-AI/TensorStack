// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.IO;
using TensorStack.Common;
using TensorStack.StableDiffusion.Enums;
using TensorStack.StableDiffusion.Pipelines.StableDiffusion;

namespace TensorStack.StableDiffusion.Pipelines.LatentConsistency
{
    public record LatentConsistencyConfig : StableDiffusionConfig
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="LatentConsistencyConfig"/> class.
        /// </summary>
        public LatentConsistencyConfig() : base()
        {
            Name = "LatentConsistency";
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
        /// <returns>StableDiffusionConfig.</returns>
        public static new LatentConsistencyConfig FromDefault(string name, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            var config = new LatentConsistencyConfig { Name = name };
            config.Unet.ModelType = modelType;
            config.SetProvider(executionProvider);
            return config;
        }


        /// <summary>
        /// Create LatentConsistency configuration from json file
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>LatentConsistencyConfig.</returns>
        public static new LatentConsistencyConfig FromFile(string configFile, ExecutionProvider executionProvider = default)
        {
            var config = ConfigService.Deserialize<LatentConsistencyConfig>(configFile);
            config.SetProvider(executionProvider);
            return config;
        }


        /// <summary>
        /// Create LatentConsistency configuration from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>LatentConsistencyConfig.</returns>
        public static new LatentConsistencyConfig FromFolder(string modelFolder, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            var config = FromDefault(Path.GetFileNameWithoutExtension(modelFolder), modelType, executionProvider);
            config.Tokenizer.Path = Path.Combine(modelFolder, "tokenizer", "vocab.json");
            config.TextEncoder.Path = Path.Combine(modelFolder, "text_encoder", "model.onnx");
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
