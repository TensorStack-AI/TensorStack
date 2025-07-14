// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common;
using TensorStack.StableDiffusion.Config;

namespace TensorStack.StableDiffusion.Models
{
    /// <summary>
    /// UNetConditionalModel: StableCascade Conditional U-Net architecture to denoise the encoded image latents.
    /// </summary>
    public class StableCascadeUNet : ModelSession<UNetModelConfig>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="StableCascadeUNet"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public StableCascadeUNet(UNetModelConfig configuration)
            : base(configuration) { }

    }
}
