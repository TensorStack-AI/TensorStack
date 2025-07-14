// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.IO;
using TensorStack.Common;
using TensorStack.Extractors.Common;

namespace TensorStack.Extractors.Models
{
    /// <summary>
    /// Default Extractor ModelSession.
    /// </summary>
    public class ExtractorModel : ModelSession<ExtractorConfig>
    {
        private ExtractorModel(ExtractorConfig configuration)
            : base(configuration) { }

        /// <summary>
        /// The channels the model supports 1 = Greyscale, RGB = 3, RGBA = 4.
        /// </summary>
        public int Channels => Configuration.Channels;

        /// <summary>
        /// The models input maximum size (0 = Any)
        /// </summary>
        public int SampleSize => Configuration.SampleSize;

        /// <summary>
        /// The models expected input normalization (0-1 or -1-1)
        /// </summary>
        public Normalization Normalization => Configuration.Normalization;

        /// <summary>
        /// The models required output normalization
        /// </summary>
        public Normalization OutputNormalization => Configuration.OutputNormalization;

        /// <summary>
        /// If the result should ne inverted
        /// </summary>
        public bool IsOutputInverted => Configuration.IsOutputInverted;

        /// <summary>
        /// The channels the model supports 1 = Greyscale, RGB = 3, RGBA = 4.
        /// </summary>
        public int OutputChannels => Configuration.OutputChannels;


        /// <summary>
        /// Create a ExtractorModel with the specified ExtractorConfig
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        /// <returns>ExtractorModel.</returns>
        /// <exception cref="System.IO.FileNotFoundException">ExtractorModel not found</exception>
        public static ExtractorModel Create(ExtractorConfig configuration)
        {
            if (!File.Exists(configuration.Path))
                throw new FileNotFoundException("ExtractorModel not found", configuration.Path);

            return new ExtractorModel(configuration);
        }
    }
}
