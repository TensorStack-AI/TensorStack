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
    /// TransformerModel: Wan Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
    /// </summary>
    public class TransformerWanModel : TransformerModel
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TransformerWanModel"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public TransformerWanModel(TransformerModelConfig configuration)
            : base(configuration) { }


        /// <summary>
        /// Runs the Transformer model with the specified inputs
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <param name="hiddenStates">The hidden states.</param>
        /// <param name="encoderHiddenStates">The encoder hidden states.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task<Tensor<float>> RunAsync(int timestep, Tensor<float> hiddenStates, Tensor<float> encoderHiddenStates, CancellationToken cancellationToken = default)
        {
            if (!Transformer.IsLoaded())
                await Transformer.LoadAsync(cancellationToken: cancellationToken);

            using (var transformerParams = new ModelParameters(Transformer.Metadata, cancellationToken))
            {
                // Inputs
                transformerParams.AddInput(hiddenStates);
                transformerParams.AddScalarInput(timestep);
                transformerParams.AddInput(encoderHiddenStates);

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
