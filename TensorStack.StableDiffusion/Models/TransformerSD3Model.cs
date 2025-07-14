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
    /// TransformerModel: SD3 Conditional Transformer (MMDiT) architecture to denoise the encoded image latents.
    /// </summary>
    public class TransformerSD3Model : TransformerModel
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="TransformerSD3Model"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public TransformerSD3Model(TransformerModelConfig configuration)
            : base(configuration) { }


        /// <summary>
        /// Runs the Transformer model with the specified inputs
        /// </summary>
        /// <param name="timestep">The timestep.</param>
        /// <param name="hiddenStates">The hidden states.</param>
        /// <param name="encoderHiddenStates">The encoder hidden states.</param>
        /// <param name="pooledProjections">The pooled projections.</param>
        /// <param name="outputDimension">The output dimension.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task<Tensor<float>> RunAsync(int timestep, Tensor<float> hiddenStates, Tensor<float> encoderHiddenStates, Tensor<float> pooledProjections, CancellationToken cancellationToken = default)
        {
            if (TransformerControlNet.IsLoaded())
                await TransformerControlNet.UnloadAsync();
            if (!Transformer.IsLoaded())
                await Transformer.LoadAsync(cancellationToken: cancellationToken);

            var timestepIndex = Transformer.Metadata.Inputs.IndexOf(x => x.Dimensions.Length == 1);
            using (var transformerParams = new ModelParameters(Transformer.Metadata, cancellationToken))
            {
                // Inputs
                transformerParams.AddInput(hiddenStates.AsTensorSpan());
                if (timestepIndex == 1)
                    transformerParams.AddScalarInput(timestep);
                transformerParams.AddInput(encoderHiddenStates.AsTensorSpan());
                transformerParams.AddInput(pooledProjections.AsTensorSpan());
                if (timestepIndex > 1)
                    transformerParams.AddScalarInput(timestep);

                // Outputs
                transformerParams.AddOutput(hiddenStates.Dimensions);

                // Inference
                using (var results = await Transformer.RunInferenceAsync(transformerParams))
                {
                    return results[0].ToTensor();
                }
            }
        }


        /// <summary>
        /// Runs the Transformer + ControlNet model with the specified inputs
        /// </summary>
        /// <param name="controlNet">The control net.</param>
        /// <param name="controlSample">The control sample.</param>
        /// <param name="conditioningScale">The conditioning scale.</param>
        /// <param name="timestep">The timestep.</param>
        /// <param name="hiddenStates">The hidden states.</param>
        /// <param name="encoderHiddenStates">The encoder hidden states.</param>
        /// <param name="pooledProjections">The pooled projections.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task<Tensor<float>> RunAsync(ControlNetModel controlNet, Tensor<float> controlSample, float conditioningScale, int timestep, Tensor<float> hiddenStates, Tensor<float> encoderHiddenStates, Tensor<float> pooledProjections, CancellationToken cancellationToken = default)
        {
            if (Transformer.IsLoaded())
                await Transformer.UnloadAsync();
            if (!TransformerControlNet.IsLoaded())
                await TransformerControlNet.LoadAsync(cancellationToken: cancellationToken);
            if (!controlNet.IsLoaded())
                await controlNet.LoadAsync(cancellationToken: cancellationToken);

            var controlNetPooledPromptEmbedsUncond = pooledProjections;
            if (controlNet.DisablePooledProjection)
            {
                // Instantx SD3 ControlNet used zero pooled projection
                controlNetPooledPromptEmbedsUncond = new Tensor<float>(pooledProjections.Dimensions);
            }

            var controlNetOutputDimension = new int[] { controlNet.LayerCount, JointAttention, CaptionProjection };
            using (var transformerParams = new ModelParameters(TransformerControlNet.Metadata, cancellationToken))
            using (var controlNetParams = new ModelParameters(controlNet.Metadata, cancellationToken))
            {
                // Transformer Inputs
                transformerParams.AddInput(hiddenStates.AsTensorSpan());
                transformerParams.AddScalarInput(timestep);
                transformerParams.AddInput(encoderHiddenStates.AsTensorSpan());
                transformerParams.AddInput(pooledProjections.AsTensorSpan());
                transformerParams.AddOutput(hiddenStates.Dimensions);

                // ControlNet Inputs
                controlNetParams.AddInput(hiddenStates.AsTensorSpan());
                controlNetParams.AddScalarInput(timestep);
                controlNetParams.AddInput(encoderHiddenStates.AsTensorSpan());
                controlNetParams.AddInput(controlNetPooledPromptEmbedsUncond.AsTensorSpan());
                controlNetParams.AddInput(controlSample.AsTensorSpan());
                controlNetParams.AddScalarInput(conditioningScale);
                controlNetParams.AddOutput(controlNetOutputDimension);

                // ControlNet Inference
               
                using (var controlNetResults = await controlNet.RunInferenceAsync(controlNetParams))
                {
                    // Transformer Inputs
                    transformerParams.AddInput(controlNetResults[0]);

                    // Transformer Inference
                    using (var transformerResults = await TransformerControlNet.RunInferenceAsync(transformerParams))
                    {
                        return transformerResults[0].ToTensor();
                    }
                }
            }
        }

    }
}
