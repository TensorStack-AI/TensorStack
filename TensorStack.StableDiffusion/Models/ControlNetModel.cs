// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common;
using TensorStack.StableDiffusion.Config;

namespace TensorStack.StableDiffusion.Models
{
    public class ControlNetModel : ModelSession<ControlNetModelConfig>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ControlNetModel"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public ControlNetModel(ControlNetModelConfig configuration)
            : base(configuration) { }

        /// <summary>
        /// Gets a value indicating whether to invert the control input.
        /// </summary>
        public bool InvertInput => Configuration.InvertInput;

        /// <summary>
        /// Gets the layer count.
        /// </summary>
        public int LayerCount => Configuration.LayerCount;

        /// <summary>
        /// Gets a value indicating whether to disable pooled projections.
        /// </summary>
        public bool DisablePooledProjection => Configuration.DisablePooledProjection;


        /// <summary>
        /// Creates the ControlNet model with the specified configuration.
        /// </summary>
        /// <param name="configFile">The configuration file.</param>
        /// <param name="executionProvider">The execution provider.</param>
        public static ControlNetModel FromConfig(string configFile, ExecutionProvider executionProvider)
        {
            var configuration = ConfigService.Deserialize<ControlNetModelConfig>(configFile);
            configuration.SetProvider(executionProvider);
            return new ControlNetModel(configuration);
        }


        /// <summary>
        /// Creates the ControlNet model with the specified provider.
        /// </summary>
        /// <param name="modelFile">The model file.</param>
        /// <param name="executionProvider">The execution provider.</param>
        /// <param name="invertInput">if set to <c>true</c> invert control input.</param>
        /// <param name="layerCount">The layer count.</param>
        /// <param name="disablePooledProjection">if set to <c>true</c> disable pooled projection.</param>
        public static ControlNetModel FromFile(string modelFile, ExecutionProvider executionProvider, bool invertInput = false, int layerCount = 0, bool disablePooledProjection = false)
        {
            var configuration = new ControlNetModelConfig
            {
                Path = modelFile,
                InvertInput = invertInput,
                LayerCount = layerCount,
                DisablePooledProjection = disablePooledProjection
            };
            configuration.SetProvider(executionProvider);
            return new ControlNetModel(configuration);
        }

    }
}
