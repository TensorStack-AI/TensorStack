
using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using TensorStack.Common;
using TensorStack.Common.Tensor;

namespace TensorStack.Image
{
    public static class Extensions
    {
        /// <summary>
        /// Converts ImageTensor to BitmapSource.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns>BitmapSource.</returns>
        public static BitmapSource ToImage(this ImageTensor imageTensor)
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
        /// Converts ImageTensorBase to ImageTensor.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns>ImageTensor.</returns>
        public static async Task<ImageInput> ToImageInputAsync(this ImageTensor imageTensor)
        {
            return await Task.Run(() => new ImageInput(imageTensor));
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
        /// Saves the specified image to file.
        /// </summary>
        /// <param name="bitmap">The bitmap.</param>
        /// <param name="filePath">The file path.</param>
        internal static void Save(this WriteableBitmap bitmap, string filePath)
        {
            var encoder = new PngBitmapEncoder();
            encoder.Frames.Add(BitmapFrame.Create(bitmap));
            using (FileStream stream = new FileStream(filePath, FileMode.Create, FileAccess.Write))
            {
                encoder.Save(stream);
            }
        }


        /// <summary>
        /// Converts WriteableBitmap to ImageTensor.
        /// </summary>
        /// <param name="writeableBitmap">The writeable bitmap.</param>
        /// <returns>ImageTensor.</returns>
        internal static ImageTensor ToTensor(this WriteableBitmap bitmapSource)
        {
            var writeableBitmap = bitmapSource.ToWriteableBitmap();
            var width = writeableBitmap.PixelWidth;
            var height = writeableBitmap.PixelHeight;
            var stride = writeableBitmap.BackBufferStride;
            var buffer = new byte[stride * height];
            writeableBitmap.CopyPixels(buffer, stride, 0);

            var hw = height * width;
            var tensor = new ImageTensor(height, width);
            var dataSpan = tensor.Memory.Span;
            var bufferSpan = buffer.AsSpan();
            for (int y = 0; y < height; y++)
            {
                int rowStart = y * stride;
                for (int x = 0; x < width; x++)
                {
                    int offset = y * width + x;
                    int pixelIndex = rowStart + x * 4; // BGRA in buffer
                    dataSpan[offset] = bufferSpan[pixelIndex + 2].NormalizeToFloat();          // R
                    dataSpan[hw + offset] = bufferSpan[pixelIndex + 1].NormalizeToFloat();     // G
                    dataSpan[2 * hw + offset] = bufferSpan[pixelIndex + 0].NormalizeToFloat(); // B
                    dataSpan[3 * hw + offset] = bufferSpan[pixelIndex + 3].NormalizeToFloat(); // A
                }
            }
            return tensor;
        }


        /// <summary>
        /// Converts ImageTensor to WriteableBitmap.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns>WriteableBitmap.</returns>
        internal static WriteableBitmap ToBitmapImage(this ImageTensor imageTensor)
        {
            var height = imageTensor.Dimensions[2];
            var width = imageTensor.Dimensions[3];
            var stride = width * 4;
            var pixelBuffer = new byte[height * stride];
            var writeableBitmap = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgra32, null);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int pixelIndex = (y * width + x) * 4;
                    pixelBuffer[pixelIndex + 0] = imageTensor[0, 2, y, x].DenormalizeToByte(); // B
                    pixelBuffer[pixelIndex + 1] = imageTensor[0, 1, y, x].DenormalizeToByte(); // G
                    pixelBuffer[pixelIndex + 2] = imageTensor[0, 0, y, x].DenormalizeToByte(); // R
                    pixelBuffer[pixelIndex + 3] = imageTensor[0, 3, y, x].DenormalizeToByte(); // A
                }
            }
            writeableBitmap.WritePixels(new Int32Rect(0, 0, width, height), pixelBuffer, stride, 0);
            writeableBitmap.Freeze();
            return writeableBitmap;
        }


        internal static WriteableBitmap ToWriteableBitmap(this BitmapSource bitmapSource)
        {
            if (bitmapSource.Format == PixelFormats.Bgra32 || bitmapSource.Format == PixelFormats.Bgr32)
            {
                if (bitmapSource is WriteableBitmap writeableBitmap)
                    return writeableBitmap;

                writeableBitmap = new WriteableBitmap(bitmapSource);
                writeableBitmap.Freeze();
                return writeableBitmap;
            }

            // Convert to BGRA32 WriteableBitmap
            var convertTarget = new WriteableBitmap(bitmapSource.PixelWidth, bitmapSource.PixelHeight, bitmapSource.DpiX, bitmapSource.DpiY, PixelFormats.Bgra32, null);
            var stride = convertTarget.PixelWidth * (convertTarget.Format.BitsPerPixel / 8);
            var buffer = new byte[stride * convertTarget.PixelHeight];
            var convertSource = new FormatConvertedBitmap(bitmapSource, PixelFormats.Bgra32, null, 0);
            convertSource.CopyPixels(buffer, stride, 0);
            convertTarget.WritePixels(new Int32Rect(0, 0, convertTarget.PixelWidth, convertTarget.PixelHeight), buffer, stride, 0);
            convertTarget.Freeze();
            return convertTarget;
        }

    }
}
