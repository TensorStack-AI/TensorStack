// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.IO;
using TensorStack.Common;

namespace TensorStack.Video.Models
{
    public class InterpolationModel : ModelSession<ModelConfig>
    {
        private InterpolationModel(ModelConfig configuration)
            : base(configuration) { }

        public int Channels { get; init; } = 3;
        public Normalization Normalization { get; init; } = Normalization.ZeroToOne;
        public Normalization OutputNormalization { get; init; } = Normalization.OneToOne;

        public static InterpolationModel Create(ModelConfig configuration)
        {
            if (!File.Exists(configuration.Path))
                throw new FileNotFoundException("InterpolationModel not found", configuration.Path);

            return new InterpolationModel(configuration);
        }
    }
}
