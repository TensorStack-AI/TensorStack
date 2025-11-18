// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.IO;
using TensorStack.Common;
using TensorStack.StableDiffusion.Enums;

namespace TensorStack.StableDiffusion.Pipelines
{
    public abstract record PipelineConfig
    {
        /// <summary>
        /// Gets or sets the type.
        /// </summary>
        public abstract PipelineType Pipeline { get;}

        /// <summary>
        /// Saves the configuration to file.
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="useRelativePaths">if set to <c>true</c> use relative paths.</param>
        public abstract void Save(string configFile, bool useRelativePaths = true);

        /// <summary>
        /// Sets the execution provider for all models.
        /// </summary>
        /// <param name="executionProvider">The execution provider.</param>
        public abstract void SetProvider(ExecutionProvider executionProvider);

        /// <summary>
        /// Gets the variant path if it exists.
        /// </summary>
        /// <param name="modelFolder">The model folder.</param>
        /// <param name="model">The model.</param>
        /// <param name="variant">The variant.</param>
        /// <param name="filename">The filename.</param>
        protected static string GetVariantPath(string modelFolder, string model, string filename, string variant = default)
        {
            if (!string.IsNullOrEmpty(variant))
            {
                var variantPath = Path.Combine(modelFolder, model, variant, filename);
                if (File.Exists(variantPath))
                    return variantPath;
            }

            return Path.Combine(modelFolder, model, filename);
        }
    }
}
