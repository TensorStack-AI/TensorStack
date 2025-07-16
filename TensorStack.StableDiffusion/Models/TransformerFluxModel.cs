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
    /// TransformerModel: FLUX Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
    /// </summary>
    public class TransformerFluxModel : TransformerModel
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TransformerFluxModel"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public TransformerFluxModel(TransformerModelConfig configuration)
            : base(configuration) { }


        /// <summary>
        /// Runs the Transformer model with the specified inputs
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <param name="hiddenStates">The hidden states.</param>
        /// <param name="encoderHiddenStates">The encoder hidden states.</param>
        /// <param name="pooledProjections">The pooled projections.</param>
        /// <param name="imgIds">The img ids.</param>
        /// <param name="txtIds">The text ids.</param>
        /// <param name="guidanceTensor">The guidance tensor.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task<Tensor<float>> RunAsync(int timestep, Tensor<float> hiddenStates, Tensor<float> encoderHiddenStates, Tensor<float> pooledProjections, Tensor<float> imgIds, Tensor<float> txtIds, float guidanceScale, CancellationToken cancellationToken = default)
        {
            if (!Transformer.IsLoaded())
                await Transformer.LoadAsync(cancellationToken: cancellationToken);

            var supportsGuidance = Transformer.Metadata.Inputs.Count == 7;
            using (var transformerParams = new ModelParameters(Transformer.Metadata, cancellationToken))
            {
                // Inputs
                transformerParams.AddInput(hiddenStates.AsTensorSpan());
                transformerParams.AddInput(encoderHiddenStates.AsTensorSpan());
                transformerParams.AddInput(pooledProjections.AsTensorSpan());
                transformerParams.AddScalarInput(timestep / 1000f);
                transformerParams.AddInput(imgIds.AsTensorSpan());
                transformerParams.AddInput(txtIds.AsTensorSpan());
                if (supportsGuidance)
                    transformerParams.AddScalarInput(guidanceScale);

                // Outputs
                transformerParams.AddOutput(hiddenStates.Dimensions);

                // Inference
                using (var results = await Transformer.RunInferenceAsync(transformerParams))
                {
                    return results[0].ToTensor();
                }
            }
        }

    }
}
