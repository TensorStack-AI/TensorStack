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
    /// CLIPVisionModelWithProjection: Frozen CLIP image-encoder ([clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14)).
    /// </summary>
    public class CLIPVisionModelWithProjection : ModelSession<CLIPModelConfig>
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="CLIPVisionModelWithProjection"/> class.
        /// </summary>
        /// <param name="configuration">The configuration.</param>
        public CLIPVisionModelWithProjection(CLIPModelConfig configuration)
            : base(configuration) { }

        /// <summary>
        /// Gets the hidden size.
        /// </summary>
        public int HiddenSize => Configuration.HiddenSize;


        /// <summary>
        /// Run the model inference with the specified input
        /// </summary>
        /// <param name="inputTensor">The input tensor.</param>
        /// <param name="cancellationToken">The cancellation token.</param>
        public async Task<Tensor<float>> RunAsync(Tensor<float> inputTensor, CancellationToken cancellationToken = default)
        {
            if (!this.IsLoaded())
                await LoadAsync(cancellationToken: cancellationToken);

            using (var modelParameters = new ModelParameters(Metadata, cancellationToken))
            {
                // Inputs
                modelParameters.AddInput(inputTensor.AsTensorSpan());

                // Outputs
                modelParameters.AddOutput([1, HiddenSize]);

                // Inference
                using (var results = await RunInferenceAsync(modelParameters))
                {
                    return results[0].ToTensor();
                }
            }
        }

    }
}
