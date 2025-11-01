// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Config;

namespace TensorStack.StableDiffusion.Models
{
    public class FlowEstimationModel :ModelSession<FlowEstimationModelConfig>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="FlowEstimationModel"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public FlowEstimationModel(FlowEstimationModelConfig configuration)
            : base(configuration) { }


        /// <summary>
        /// Runs the model with the specified input
        /// </summary>
        /// <param name="frameTensor">The frame tensor.</param>
        /// <param name="previousFrameTensor">The previous frame tensor.</param>
        /// <param name="timestep">The timestep.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task<Tensor<float>> RunAsync(Tensor<float> frameTensor, Tensor<float> previousFrameTensor, float timestep, CancellationToken cancellationToken = default)
        {
            if (!this.IsLoaded())
                await LoadAsync(cancellationToken: cancellationToken);

            using (var modelParameters = new ModelParameters(Metadata))
            {
                // Inputs
                modelParameters.AddInput(previousFrameTensor.AsTensorSpan());
                modelParameters.AddInput(frameTensor.AsTensorSpan());
                modelParameters.AddScalarInput(timestep);

                // Outputs
                modelParameters.AddOutput(frameTensor.Dimensions);

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
