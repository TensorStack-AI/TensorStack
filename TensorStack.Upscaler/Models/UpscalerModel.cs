// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using Microsoft.ML.OnnxRuntime;
using System;
using System.IO;
using TensorStack.Common;
using TensorStack.Core.Inference;
using TensorStack.Upscaler.Common;

namespace TensorStack.Upscaler.Models
{
    /// <summary>
    /// Default Upscale ModelSession.
    /// </summary>
    /// <seealso cref="TensorStack.Core.Inference.ModelSession{UpscalerConfig}" />
    public class UpscalerModel : ModelSession<UpscalerConfig>
    {
        private UpscalerModel(UpscalerConfig configuration)
            : base(configuration) { }

        private UpscalerModel(UpscalerConfig configuration, Func<SessionOptions> sessionOptionsFactory)
            : base(configuration, sessionOptionsFactory) { }

        /// <summary>
        /// The channels the model supports RGB = 3, RGBA = 4.
        /// </summary>
        public int Channels => Configuration.Channels;

        /// <summary>
        /// The models input size 
        /// </summary>
        public int SampleSize => Configuration.SampleSize;

        /// <summary>
        /// The scale factor the model supports, 2x 4x etc
        /// </summary>
        public int ScaleFactor => Configuration.ScaleFactor;

        /// <summary>
        /// The models expected input normalization (0-1 or -1-1)
        /// </summary>
        /// <value>The normalization.</value>
        public Normalization Normalization => Configuration.Normalization;


        /// <summary>
        /// Create a UpscalerModel with the specified UpscalerConfig
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <returns>UpscalerModel.</returns>
        /// <exception cref="System.IO.FileNotFoundException">UpscalerModel not found</exception>
        public static UpscalerModel Create(UpscalerConfig configuration)
        {
            if (!File.Exists(configuration.Path))
                throw new FileNotFoundException("UpscalerModel not found", configuration.Path);

            return new UpscalerModel(configuration);
        }


        /// <summary>
        /// Create a UpscalerModel with the specified UpscalerConfig and SessionOptionsFactory
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <param name="sessionOptionsFactory">The session options factory.</param>
        /// <returns>UpscalerModel.</returns>
        /// <exception cref="System.IO.FileNotFoundException">UpscalerModel not found</exception>
        public static UpscalerModel Create(UpscalerConfig configuration, Func<SessionOptions> sessionOptionsFactory)
        {
            if (!File.Exists(configuration.Path))
                throw new FileNotFoundException("UpscalerModel not found", configuration.Path);

            return new UpscalerModel(configuration, sessionOptionsFactory);
        }
    }
}
