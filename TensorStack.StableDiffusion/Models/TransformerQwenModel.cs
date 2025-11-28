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
    /// TransformerModel: QwenImageTransformer2DModel
    /// </summary>
    public class TransformerQwenModel : TransformerModel
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TransformerQwenModel"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public TransformerQwenModel(TransformerModelConfig configuration)
            : base(configuration) { }


        /// <summary>
        /// Runs the Transformer model with the specified inputs
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <param name="hiddenStates">The hidden states.</param>
        /// <param name="encoderHiddenStates">The encoder hidden states.</param>
        /// <param name="imgShapes">The image shapes.</param>
        /// <param name="cancellationToken">The cancellation token that can be used by other objects or threads to receive notice of cancellation.</param>
        public async Task<Tensor<float>> RunAsync(int timestep, Tensor<float> hiddenStates, Tensor<float> encoderHiddenStates, Tensor<long> imgShapes, CancellationToken cancellationToken = default)
        {
            if (!Transformer.IsLoaded())
                await Transformer.LoadAsync(cancellationToken: cancellationToken);

            using (var transformerParams = new ModelParameters(Transformer.Metadata, cancellationToken))
            {
                // Inputs
                transformerParams.AddInput(hiddenStates);
                transformerParams.AddInput(encoderHiddenStates);
                transformerParams.AddScalarInput(timestep);
                transformerParams.AddInput(imgShapes);

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
