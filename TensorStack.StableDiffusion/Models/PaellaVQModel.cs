// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Config;

namespace TensorStack.StableDiffusion.Models
{
    /// <summary>
    /// PaellaVQModel: VQ-VAE model from Paella model
    /// </summary>
    public class PaellaVQModel : ModelSession<PaellaVQModelConfig>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="PaellaVQModel"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public PaellaVQModel(PaellaVQModelConfig configuration)
            : base(configuration) { }

        /// <summary>
        /// Gets the scale.
        /// </summary>
        public int Scale => Configuration.Scale;

        /// <summary>
        /// Gets the output channels.
        /// </summary>
        public int OutChannels => Configuration.OutChannels;

        /// <summary>
        /// Gets the scale factor.
        /// </summary>
        public float ScaleFactor => Configuration.ScaleFactor;

        /// <summary>
        /// Gets the shift factor.
        /// </summary>
        public float ShiftFactor => Configuration.ShiftFactor;


        /// <summary>
        /// Runs the model with the specified input
        /// </summary>
        /// <param name="inputTensor">The input tensor.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public virtual async Task<Tensor<float>> RunAsync(Tensor<float> inputTensor, bool disableShift = false, bool disableScale = false, CancellationToken cancellationToken = default)
        {
            if (!this.IsLoaded())
                await LoadAsync(cancellationToken: cancellationToken);

            if (!disableScale)
                inputTensor.Multiply(ScaleFactor);
            if (!disableShift)
                inputTensor.Add(ShiftFactor);

            var outputDimensions = new[] { 1, OutChannels, inputTensor.Dimensions[2] * Scale, inputTensor.Dimensions[3] * Scale };
            using (var modelParameters = new ModelParameters(Metadata, cancellationToken))
            {
                // Inputs
                modelParameters.AddInput(inputTensor.AsTensorSpan());

                // Outputs
                modelParameters.AddOutput(outputDimensions);

                // Inference
                using (var results = await RunInferenceAsync(modelParameters))
                {
                    return results[0]
                        .ToTensor()
                        .NormalizeOneOne();
                }
            }
        }

    }
}
