// Copyright (c) 2026 Joe Dluzen. All rights reserved.
// Licensed under the Apache 2.0 License.

using Microsoft.ML.OnnxRuntime.Tensors;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;
using TensorStack.StableDiffusion.Config;
using TensorStack.StableDiffusion.Models;

namespace TensorStack.StableDiffusion.Pipelines.Nitro
{
    /// <summary>
    /// TransformerModel: Nitro-AR Autoregressive Transformer used to predict and unmask continuous image tokens.
    /// </summary>
    public class TransformerNitroARModel : TransformerModel
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TransformerNitroARModel"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public TransformerNitroARModel(TransformerModelConfig configuration)
            : base(configuration) { }

        /// <summary>
        /// Runs the Nitro-AR Transformer model with the specified inputs.
        /// </summary>
        /// <param name="timestep">The dummy timestep (usually 0f for AR models).</param>
        /// <param name="hiddenStates">The masked latent canvas.</param>
        /// <param name="encoderHiddenStates">The text prompt embeddings.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        /// <returns>A Task&lt;Tensor`1&gt; representing the asynchronous operation.</returns>
        public async Task<TensorStack.Common.Tensor.Tensor<float>> RunAsync(TensorStack.Common.Tensor.Tensor<float> hiddenStates, TensorStack.Common.Tensor.Tensor<float> encoderHiddenStates, CancellationToken cancellationToken = default)
        {
            if (!Transformer.IsLoaded())
                await Transformer.LoadAsync(cancellationToken: cancellationToken);

            using (var transformerParams = new ModelParameters(Transformer.Metadata, cancellationToken))
            {
                int batchSize = hiddenStates.Dimensions[0];

                transformerParams.AddInput(hiddenStates.AsTensorSpan());
                transformerParams.AddInput(encoderHiddenStates.AsTensorSpan());

                transformerParams.AddOutput(hiddenStates.Dimensions);

                using (var results = await Transformer.RunInferenceAsync(transformerParams))
                {
                    return results[0].ToTensor();
                }
            }
        }
    }
}