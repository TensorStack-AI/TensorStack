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
            return CreateFromFolder(modelFolder, default, modelType, executionProvider);
        }


        /// <summary>
        /// Create LatentConsistency configuration from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>LatentConsistencyConfig.</returns>
        public static new LatentConsistencyConfig FromFolder(string modelFolder, string variant, ModelType modelType, ExecutionProvider executionProvider = default)
        {
            return CreateFromFolder(modelFolder, variant, modelType, executionProvider);
        }


        /// <summary>
        /// Create LatentConsistency configuration from folder structure
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="modelType">Type of the model.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <returns>LatentConsistencyConfig.</returns>
        private static LatentConsistencyConfig CreateFromFolder(string modelFolder, string variant, ModelType modelType, ExecutionProvider executionProvider)
        {
            var config = FromDefault(Path.GetFileNameWithoutExtension(modelFolder), modelType, executionProvider);
            config.Tokenizer.Path = Path.Combine(modelFolder, "tokenizer", "vocab.json");
            config.TextEncoder.Path = GetVariantPath(modelFolder, "text_encoder", "model.onnx", variant);
            config.Unet.Path = GetVariantPath(modelFolder, "unet", "model.onnx", variant);
            config.AutoEncoder.DecoderModelPath = GetVariantPath(modelFolder, "vae_decoder", "model.onnx", variant);
            config.AutoEncoder.EncoderModelPath = GetVariantPath(modelFolder, "vae_encoder", "model.onnx", variant);
            var controlNetPath = GetVariantPath(modelFolder, "unet", "controlnet.onnx", variant);
            if (File.Exists(controlNetPath))
                config.Unet.ControlNetPath = controlNetPath;
            return config;
        }
    }
}
