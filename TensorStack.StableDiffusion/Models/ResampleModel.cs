// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Common;
using TensorStack.StableDiffusion.Config;

namespace TensorStack.StableDiffusion.Models
{
    public class ResampleModel : ModelSession<ResampleModelConfig>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ResampleModel"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public ResampleModel(ResampleModelConfig configuration)
            : base(configuration) { }


        public int ScaleFactor => Configuration.ScaleFactor;


        /// <summary>
        /// Runs the model with the specified input
        /// </summary>
        /// <param name="options">The options.</param>
        /// <param name="inputTensor">The input tensor.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task<Tensor<float>> RunAsync(GenerateOptions options, Tensor<float> inputTensor, CancellationToken cancellationToken = default)
        {
            if (!this.IsLoaded())
                await LoadAsync(cancellationToken: cancellationToken);

            var upsampleWidth = options.Width * ScaleFactor;
            var upsampleHeight = options.Height * ScaleFactor;
            var outputDimensions = new[] { 1, 3, upsampleHeight, upsampleWidth };
            using (var modelParameters = new ModelParameters(Metadata, cancellationToken))
            {
                // Inputs
                modelParameters.AddInput(inputTensor.AsTensorSpan(), Normalization.ZeroToOne);

                // Outputs
                modelParameters.AddOutput(outputDimensions);

                // Inference
                using (var results = await RunInferenceAsync(modelParameters))
                {
                    return results[0]
                        .ToTensor()
                        .Normalize(Normalization.OneToOne);
                }
            }
        }

    }
}
