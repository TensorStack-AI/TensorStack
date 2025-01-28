// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using TensorStack.Common.Tensor;

namespace TensorStack.Common.Vision
{
    /// <summary>
    /// CLIPImage class for preprocessing images to prepare them for use with a CLIP model
    /// </summary>
    public class CLIPImage
    {
        /// <summary>
        /// Resizes and normalizes an ImageTensor for CLIP input.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <returns>ImageTensor.</returns>
        public static ImageTensor Process(ImageTensor input, int width = 224, int height = 224)
        {
            var options = new CLIPImageOptions(width, height);
            return Process(input, options);
        }


        /// <summary>
        /// Resizes and normalizes an ImageTensor for CLIP input.
        /// </summary>
        /// <param name="input">The input.</param>
        /// <param name="options">The options.</param>
        /// <returns>ImageTensor.</returns>
        public static ImageTensor Process(ImageTensor input, CLIPImageOptions options)
        {
            options ??= new CLIPImageOptions();
            var resultTensor = input.ResizeImage(options.Width, options.Height, ResizeMode.Stretch);
            resultTensor.NormalizeZeroToOne();
            for (int x = 0; x < resultTensor.Width; x++)
            {
                for (int y = 0; y < resultTensor.Height; y++)
                {
                    resultTensor[0, 0, y, x] = (resultTensor[0, 0, y, x] - options.Mean[0]) / options.StdDev[0];
                    resultTensor[0, 1, y, x] = (resultTensor[0, 1, y, x] - options.Mean[1]) / options.StdDev[1];
                    resultTensor[0, 2, y, x] = (resultTensor[0, 2, y, x] - options.Mean[2]) / options.StdDev[2];
                }
            }
            return resultTensor;
        }
    }
}
