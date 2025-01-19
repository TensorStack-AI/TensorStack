// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using System;
using System.Threading.Tasks;
using TensorStack.Common.Image;
using TensorStack.Common.Tensor;

namespace TensorStack.Image
{
    public static class Extensions
    {
        /// <summary>
        /// Converts ImageTensorBase to ImageTensor.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns>ImageTensor.</returns>
        public static ImageInput ToImageTensor(this ImageTensor imageTensor)
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
                using (var image = imageTensor.ToImageTensor())
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
        /// Resizes the specified image.
        /// </summary>
        /// <param name="image">The image.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="resizeMode">The resize mode.</param>
        internal static Image<Rgba32> Resize(this Image<Rgba32> image, int width, int height, Common.ResizeMode resizeMode = Common.ResizeMode.Stretch)
        {
            image.Mutate(x =>
            {
                x.Resize(new ResizeOptions
                {
                    Size = new Size(width, height),
                    Mode = GetResizeMode(resizeMode),
                    Sampler = KnownResamplers.Lanczos8,
                    Compand = true
                });
            });
            return image;
        }


        /// <summary>
        /// Creates a CLIP feature tensor.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        internal static ImageTensor CreateClipFeatureTensor(this ImageInput imageTensor, ImageClipOptions clipOptions = default)
        {
            clipOptions ??= new ImageClipOptions();
            var image = imageTensor.Image.Clone();
            image.Resize(clipOptions.Width, clipOptions.Height);
            var imageArray = new Tensor<float>([1, 3, clipOptions.Height, clipOptions.Width]);
            image.ProcessPixelRows(img =>
            {
                for (int x = 0; x < image.Width; x++)
                {
                    for (int y = 0; y < image.Height; y++)
                    {
                        var pixelSpan = img.GetRowSpan(y);
                        imageArray[0, 0, y, x] = ((pixelSpan[x].R / 255f) - clipOptions.Mean[0]) / clipOptions.StdDev[0];
                        imageArray[0, 1, y, x] = ((pixelSpan[x].G / 255f) - clipOptions.Mean[1]) / clipOptions.StdDev[1];
                        imageArray[0, 2, y, x] = ((pixelSpan[x].B / 255f) - clipOptions.Mean[2]) / clipOptions.StdDev[2];
                    }
                }
            });
            return new ImageTensor(imageArray);
        }


        /// <summary>
        /// Converts ImageSharp to Tensor.
        /// </summary>
        /// <param name="image">The image.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        internal static ImageTensor ToTensor(this Image<Rgba32> image)
        {
            var height = image.Height;
            var width = image.Width;
            var imageArray = new Tensor<float>(new[] { 1, 4, height, width });
            image.ProcessPixelRows(img =>
            {
                for (int x = 0; x < width; x++)
                {
                    for (int y = 0; y < height; y++)
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
        internal static Image<Rgba32> ToImage(this ImageTensor imageTensor)
        {
            var height = imageTensor.Dimensions[2];
            var width = imageTensor.Dimensions[3];
            var channels = imageTensor.Dimensions[1];
            if (channels == 1)
                return imageTensor.ToSingleChannelImage();

            var imageData = new Image<Rgba32>(width, height);
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    imageData[x, y] = new Rgba32
                    (
                        GetByteValue(imageTensor[0, 0, y, x]),
                        GetByteValue(imageTensor[0, 1, y, x]),
                        GetByteValue(imageTensor[0, 2, y, x]),
                        channels == 4 ? GetByteValue(imageTensor[0, 3, y, x]) : byte.MaxValue
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
            var width = imageTensor.Dimensions[3];
            var height = imageTensor.Dimensions[2];
            using (var result = new Image<L8>(width, height))
            {
                for (var y = 0; y < height; y++)
                {
                    for (var x = 0; x < width; x++)
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


        /// <summary>
        /// Gets the ImageSharp resize mode.
        /// </summary>
        /// <param name="resizeMode">The resize mode.</param>
        /// <returns>ResizeMode.</returns>
        private static ResizeMode GetResizeMode(Common.ResizeMode resizeMode)
        {
            return resizeMode switch
            {
                Common.ResizeMode.Stretch => ResizeMode.Stretch,
                _ => ResizeMode.Crop
            };
        }
    }
}
