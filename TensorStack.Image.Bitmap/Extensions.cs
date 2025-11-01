// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Drawing;
using System.Drawing.Imaging;
using System.Threading.Tasks;
using TensorStack.Common;
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
            var tensor = new ImageTensor(height, width);
            var bitmapData = bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.ReadOnly, PixelFormat.Format32bppArgb);
            unsafe
            {
                for (int y = 0; y < height; y++)
                {
                    byte* pixelRow = (byte*)bitmapData.Scan0 + (y * bitmapData.Stride);
                    for (int x = 0; x < width; x++)
                    {
                        tensor[0, 3, y, x] = pixelRow[x * 4 + 3].NormalizeToFloat(); // A
                        tensor[0, 0, y, x] = pixelRow[x * 4 + 2].NormalizeToFloat(); // R
                        tensor[0, 1, y, x] = pixelRow[x * 4 + 1].NormalizeToFloat(); // G
                        tensor[0, 2, y, x] = pixelRow[x * 4 + 0].NormalizeToFloat(); // B
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
            var bitmap = new Bitmap(width, height, PixelFormat.Format32bppArgb);
            var bitmapData = bitmap.LockBits(new Rectangle(0, 0, width, height), ImageLockMode.WriteOnly, PixelFormat.Format32bppArgb);
            unsafe
            {
                for (int y = 0; y < height; y++)
                {
                    byte* pixelRow = (byte*)bitmapData.Scan0 + (y * bitmapData.Stride);
                    for (int x = 0; x < width; x++)
                    {
                        pixelRow[x * 4 + 0] = tensor[0, 2, y, x].DenormalizeToByte(); // B
                        pixelRow[x * 4 + 1] = tensor[0, 1, y, x].DenormalizeToByte(); // G
                        pixelRow[x * 4 + 2] = tensor[0, 0, y, x].DenormalizeToByte(); // R
                        pixelRow[x * 4 + 3] = tensor[0, 3, y, x].DenormalizeToByte(); // A
                    }
                }
            }
            bitmap.UnlockBits(bitmapData);
            return bitmap;
        }

    }
}
