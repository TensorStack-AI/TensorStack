// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System;
using System.Threading.Tasks;
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
            var imageArray = new Tensor<float>(new[] { 1, 4, image.Height, image.Width });
            image.ProcessPixelRows(img =>
            {
                for (int x = 0; x < image.Width; x++)
                {
                    for (int y = 0; y < image.Height; y++)
                    {
                        var pixelSpan = img.GetRowSpan(y);
                        imageArray[0, 0, y, x] = GetFloatValue(pixelSpan[x].R);
                        imageArray[0, 1, y, x] = GetFloatValue(pixelSpan[x].G);
                        imageArray[0, 2, y, x] = GetFloatValue(pixelSpan[x].B);
                        imageArray[0, 3, y, x] = GetFloatValue(pixelSpan[x].A);
                    }
                }
            });
            return new ImageTensor(imageArray);
        }


        /// <summary>
        /// Converts Tensor to ImageSharp.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns>Image&lt;Rgba32&gt;.</returns>
        internal static Image<Rgba32> ToImageSharp(this ImageTensor imageTensor)
        {
            if (imageTensor.Channels == 1)
                return imageTensor.ToSingleChannelImage();

            var imageData = new Image<Rgba32>(imageTensor.Width, imageTensor.Height);
            for (var y = 0; y < imageTensor.Height; y++)
            {
                for (var x = 0; x < imageTensor.Width; x++)
                {
                    imageData[x, y] = new Rgba32
                    (
                        GetByteValue(imageTensor[0, 0, y, x]),
                        GetByteValue(imageTensor[0, 1, y, x]),
                        GetByteValue(imageTensor[0, 2, y, x]),
                        imageTensor.Channels == 4 ? GetByteValue(imageTensor[0, 3, y, x]) : byte.MaxValue
                    );
                }
            }
            return imageData;
        }


        /// <summary>
        /// Converts to single channel Image.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns>Image&lt;Rgba32&gt;.</returns>
        private static Image<Rgba32> ToSingleChannelImage(this ImageTensor imageTensor)
        {
            using (var result = new Image<L8>(imageTensor.Width, imageTensor.Height))
            {
                for (var y = 0; y < imageTensor.Height; y++)
                {
                    for (var x = 0; x < imageTensor.Width; x++)
                    {
                        result[x, y] = new L8((byte)(imageTensor[0, 0, y, x] * 255.0f));
                    }
                }
                return result.CloneAs<Rgba32>();
            }
        }


        /// <summary>
        /// Gets the normalized byte value.
        /// </summary>
        /// <param name="value">The value.</param>
        private static byte GetByteValue(float value)
        {
            return (byte)Math.Round(Math.Clamp(value / 2 + 0.5, 0, 1) * 255);
        }


        /// <summary>
        /// Gets the normalized float value.
        /// </summary>
        /// <param name="value">The value.</param>
        private static float GetFloatValue(byte value)
        {
            return (value / 255.0f) * 2.0f - 1.0f;
        }

    }
}
