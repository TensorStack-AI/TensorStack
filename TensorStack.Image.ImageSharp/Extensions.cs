// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Threading.Tasks;
using TensorStack.Common;
using TensorStack.Common.Tensor;

namespace TensorStack.Image
{
    public static class Extensions
    {
        /// <summary>
        /// Converts ImageTensor to ImageSharp.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns>Image&lt;Rgba32&gt;.</returns>
        public static Image<Rgba32> ToImage(this ImageTensor imageTensor)
        {
            return imageTensor.ToImageSharp();
        }

        /// <summary>
        /// Converts ImageTensorBase to ImageTensor.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns>ImageTensor.</returns>
        public static ImageInput ToImageInput(this ImageTensor imageTensor)
        {
            return new ImageInput(imageTensor);
        }


        /// <summary>
        /// Saves the ImageTensor.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="filename">The filename.</param>
        public static Task SaveAsync(this ImageTensor imageTensor, string filename)
        {
            return Task.Run(() =>
            {
                using (var image = imageTensor.ToImageInput())
                {
                    image.Save(filename);
                }
            });
        }


        /// <summary>
        /// Saves the ImageTensor.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="filename">The filename.</param>
        public static Task SaveAsync(this ImageInput imageTensor, string filename)
        {
            return Task.Run(() => imageTensor.Save(filename));
        }


        /// <summary>
        /// Converts ImageSharp to Tensor.
        /// </summary>
        /// <param name="image">The image.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        internal static ImageTensor ToTensor(this Image<Rgba32> image)
        {
            var imageTensor = new ImageTensor(image.Height, image.Width);
            image.ProcessPixelRows(img =>
            {
                for (int x = 0; x < image.Width; x++)
                {
                    for (int y = 0; y < image.Height; y++)
                    {
                        var pixelSpan = img.GetRowSpan(y);
                        imageTensor[0, 0, y, x] = pixelSpan[x].R.NormalizeToFloat();
                        imageTensor[0, 1, y, x] = pixelSpan[x].G.NormalizeToFloat();
                        imageTensor[0, 2, y, x] = pixelSpan[x].B.NormalizeToFloat();
                        imageTensor[0, 3, y, x] = pixelSpan[x].A.NormalizeToFloat();
                    }
                }
            });
            return imageTensor;
        }


        /// <summary>
        /// Converts Tensor to ImageSharp.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns>Image&lt;Rgba32&gt;.</returns>
        internal static Image<Rgba32> ToImageSharp(this ImageTensor imageTensor)
        {
            var imageData = new Image<Rgba32>(imageTensor.Width, imageTensor.Height);
            for (var y = 0; y < imageTensor.Height; y++)
            {
                for (var x = 0; x < imageTensor.Width; x++)
                {
                    imageData[x, y] = new Rgba32
                    (
                        imageTensor[0, 0, y, x].DenormalizeToByte(),
                        imageTensor[0, 1, y, x].DenormalizeToByte(),
                        imageTensor[0, 2, y, x].DenormalizeToByte(),
                        imageTensor[0, 3, y, x].DenormalizeToByte()
                    );
                }
            }
            return imageData;
        }

    }
}
