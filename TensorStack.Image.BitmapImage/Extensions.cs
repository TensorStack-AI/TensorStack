
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
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
        /// Converts ImageTensor to WriteableBitmap.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns>WriteableBitmap.</returns>
        internal static WriteableBitmap ToBitmapImage(this ImageTensor imageTensor)
        {
            var channels = imageTensor.Dimensions[1];
            var height = imageTensor.Dimensions[2];
            var width = imageTensor.Dimensions[3];

            if (channels == 1)
                return imageTensor.ToSingleChannelImage();

            var stride = width * 4;
            var pixelBuffer = new byte[height * stride];
            var writeableBitmap = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgra32, null);
            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    int pixelIndex = (y * width + x) * 4;
                    pixelBuffer[pixelIndex + 0] = GetByteValue(imageTensor[0, 2, y, x]); // B
                    pixelBuffer[pixelIndex + 1] = GetByteValue(imageTensor[0, 1, y, x]); // G
                    pixelBuffer[pixelIndex + 2] = GetByteValue(imageTensor[0, 0, y, x]); // R
                    pixelBuffer[pixelIndex + 3] = channels == 4 ? GetByteValue(imageTensor[0, 3, y, x]) : byte.MaxValue; // A
                }
            }
            writeableBitmap.WritePixels(new Int32Rect(0, 0, width, height), pixelBuffer, stride, 0);
            writeableBitmap.Freeze();
            return writeableBitmap;
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
            var tensor = new ImageTensor([1, 4, height, width]);
            var dataSpan = tensor.Memory.Span;
            var bufferSpan = buffer.AsSpan();
            for (int y = 0; y < height; y++)
            {
                int rowStart = y * stride;
                for (int x = 0; x < width; x++)
                {
                    int offset = y * width + x;
                    int pixelIndex = rowStart + x * 4; // BGRA in buffer
                    dataSpan[offset] = GetFloatValue(bufferSpan[pixelIndex + 2]);          // R
                    dataSpan[hw + offset] = GetFloatValue(bufferSpan[pixelIndex + 1]);     // G
                    dataSpan[2 * hw + offset] = GetFloatValue(bufferSpan[pixelIndex + 0]); // B
                    dataSpan[3 * hw + offset] = GetFloatValue(bufferSpan[pixelIndex + 3]); // A
                }
            }
            return tensor;
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


        /// <summary>
        /// Converts to single channel Image.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns>WriteableBitmap.</returns>
        private static WriteableBitmap ToSingleChannelImage(this ImageTensor imageTensor)
        {
            var width = imageTensor.Dimensions[3];
            var height = imageTensor.Dimensions[2];
            byte[] pixels = new byte[width * height];
            for (var y = 0; y < height; y++)
            {
                for (var x = 0; x < width; x++)
                {
                    pixels[y * width + x] = GetByteValue(imageTensor[0, 0, y, x]);
                }
            }

            var writeableBitmap = new WriteableBitmap(width, height, 96, 96, PixelFormats.Gray8, null);
            writeableBitmap.Lock();
            try
            {
                IntPtr buffer = writeableBitmap.BackBuffer;
                Marshal.Copy(pixels, 0, buffer, pixels.Length);
                writeableBitmap.AddDirtyRect(new Int32Rect(0, 0, width, height));
            }
            finally
            {
                writeableBitmap.Unlock();
            }
            writeableBitmap.Freeze();
            return writeableBitmap;
        }


        public static WriteableBitmap ToImageTransparent(this ImageTensor imageTensor)
        {
            int width = imageTensor.Width;
            int height = imageTensor.Height;
            int channels = imageTensor.Channels; // 1 or 4
            int stride = width * 4;

            byte[] pixels = new byte[height * stride];

            for (int y = 0; y < height; y++)
            {
                for (int x = 0; x < width; x++)
                {
                    byte alpha;
                    if (channels == 4)
                    {
                        // Last channel is alpha [-1..1]
                        float maskValue = (imageTensor[0, 3, y, x] + 1f) / 2f;
                        alpha = (byte)(maskValue * 255);
                    }
                    else
                    {
                        // Single channel → treat as grayscale alpha [-1..1]
                        float maskValue = (imageTensor[0, 0, y, x] + 1f) / 2f;
                        alpha = (byte)(maskValue * 255);
                    }

                    int offset = y * stride + x * 4;

                    if (channels == 4 && alpha > 0)
                    {
                        // Copy RGB from tensor
                        pixels[offset + 2] = GetByteValue(imageTensor[0, 0, y, x]); // R
                        pixels[offset + 1] = GetByteValue(imageTensor[0, 1, y, x]); // G
                        pixels[offset + 0] = GetByteValue(imageTensor[0, 2, y, x]); // B
                    }
                    else
                    {
                        // Transparent area or single channel → clear RGB
                        pixels[offset + 0] = 0; // B
                        pixels[offset + 1] = 0; // G
                        pixels[offset + 2] = 0; // R
                    }

                    pixels[offset + 3] = alpha; // A
                }
            }

            var bmp = new WriteableBitmap(width, height, 96, 96, PixelFormats.Bgra32, null);
            bmp.WritePixels(new Int32Rect(0, 0, width, height), pixels, stride, 0);
            bmp.Freeze();
            return bmp;
        }



        /// <summary>
        /// Gets the normalized byte value.
        /// </summary>
        /// <param name="value">The value.</param>
        private static byte GetByteValue(this float value)
        {
            return (byte)Math.Round(Math.Clamp(value / 2 + 0.5, 0, 1) * 255);
        }


        /// <summary>
        /// Gets the normalized float value.
        /// </summary>
        /// <param name="value">The value.</param>
        private static float GetFloatValue(this byte value)
        {
            return (value / 255.0f) * 2.0f - 1.0f;
        }
    }
}
