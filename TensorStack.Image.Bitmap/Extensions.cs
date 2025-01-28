// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System;
using System.Drawing;
using System.Drawing.Imaging;
using System.Threading.Tasks;
using TensorStack.Common.Tensor;

namespace TensorStack.Image
{
    public static class Extensions
    {
        /// <summary>
        /// Converts ImageTensor to Bitmap.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns>Bitmap.</returns>
        public static Bitmap ToImage(this ImageTensor imageTensor)
        {
            return imageTensor.ToBitmapImage();
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
        /// <returns>Task.</returns>
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
        /// Saves the ImageInput.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <param name="filename">The filename.</param>
        /// <returns>Task.</returns>
        public static Task SaveAsync(this ImageInput imageTensor, string filename)
        {
            return Task.Run(() => imageTensor.Save(filename));
        }


        /// <summary>
        /// Converts Bitmap to Tensor.
        /// </summary>
        /// <param name="bitmap">The bitmap.</param>
        /// <returns>Tensor&lt;System.Single&gt;.</returns>
        internal static ImageTensor ToTensor(this Bitmap bitmap)
        {
            var width = bitmap.Width;
            var height = bitmap.Height;
            bitmap.ConvertFormat(PixelFormat.Format32bppArgb);
            var tensor = new ImageTensor(new[] { 1, 4, height, width });
            var bitmapData = bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            unsafe
            {
                for (int y = 0; y < height; y++)
                {
                    byte* row = (byte*)bitmapData.Scan0 + (y * bitmapData.Stride);
                    for (int x = 0; x < width; x++)
                    {
                        tensor[0, 3, y, x] = GetFloatValue(row[x * 4 + 3]); // A
                        tensor[0, 0, y, x] = GetFloatValue(row[x * 4 + 2]); // R
                        tensor[0, 1, y, x] = GetFloatValue(row[x * 4 + 1]); // G
                        tensor[0, 2, y, x] = GetFloatValue(row[x * 4 + 0]); // B
                    }
                }
                bitmap.UnlockBits(bitmapData);
            }
            return tensor;
        }


        /// <summary>
        /// Converts Tensor to Bitmap.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        /// <returns>Bitmap.</returns>
        internal static Bitmap ToBitmapImage(this ImageTensor tensor)
        {
            var height = tensor.Dimensions[2];
            var width = tensor.Dimensions[3];
            var channels = tensor.Dimensions[1];
            var bitmap = new Bitmap(width, height, PixelFormat.Format32bppArgb);
            var bitmapData = bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
            unsafe
            {
                for (int y = 0; y < height; y++)
                {
                    byte* row = (byte*)bitmapData.Scan0 + (y * bitmapData.Stride);
                    for (int x = 0; x < width; x++)
                    {
                        row[x * 4 + 3] = channels == 4 ? GetByteValue(tensor[0, 3, y, x]) : byte.MaxValue; // A
                        row[x * 4 + 2] = GetByteValue(tensor[0, 0, y, x]); // R
                        row[x * 4 + 1] = GetByteValue(tensor[0, 1, y, x]); // G
                        row[x * 4 + 0] = GetByteValue(tensor[0, 2, y, x]); // B
                    }
                }
                bitmap.UnlockBits(bitmapData);
            }
            return bitmap;
        }


        /// <summary>
        /// Gets the normalized float value.
        /// </summary>
        /// <param name="value">The value.</param>
        private static float GetFloatValue(byte value)
        {
            return (value / 255.0f - 0.5f) * 2.0f;
        }


        /// <summary>
        /// Gets the normalized byte value.
        /// </summary>
        /// <param name="value">The value.</param>
        private static byte GetByteValue(float value)
        {
            return (byte)Math.Clamp((value / 2f + 0.5f) * 255.0f, 0, 255);
        }

    }
}
