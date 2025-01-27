
using System;
using System.IO;
using System.Runtime.InteropServices;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;
using TensorStack.Common.Image;
using TensorStack.Common.Tensor;

namespace TensorStack.Image
{
    public static class Extensions
    {
        const string RotationQuery = "System.Photo.Orientation";
        private static Dispatcher Dispatcher => Application.Current.Dispatcher;


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
        /// Loads the specified imagefile.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        /// <returns>WriteableBitmap.</returns>
        /// <exception cref="System.ArgumentNullException">filePath</exception>
        /// <exception cref="System.IO.FileNotFoundException">The file '{filePath}' does not exist.</exception>
        internal static WriteableBitmap Load(string filePath)
        {
            if (string.IsNullOrEmpty(filePath))
                throw new ArgumentNullException(nameof(filePath));
            if (!File.Exists(filePath))
                throw new FileNotFoundException($"The file '{filePath}' does not exist.", filePath);

            var imageUri = new Uri(filePath);
            var rotation = GetRotation(imageUri);
            return Dispatcher.Invoke(() =>
            {
                var image = new BitmapImage();
                image.BeginInit();
                image.Rotation = rotation;
                image.UriSource = new Uri(filePath);
                image.CacheOption = BitmapCacheOption.OnLoad;
                image.EndInit();
                image.Freeze();
                return new WriteableBitmap(image);
            });
        }


        /// <summary>
        /// Saves the specified image to file.
        /// </summary>
        /// <param name="bitmap">The bitmap.</param>
        /// <param name="filePath">The file path.</param>
        internal static void Save(this WriteableBitmap bitmap, string filePath)
        {
            Dispatcher.Invoke(() =>
            {
                var encoder = new PngBitmapEncoder();
                encoder.Frames.Add(BitmapFrame.Create(bitmap));
                using (FileStream stream = new FileStream(filePath, FileMode.Create, FileAccess.Write))
                {
                    encoder.Save(stream);
                }
            });
        }


        /// <summary>
        /// Resizes the specified image.
        /// </summary>
        /// <param name="source">The source.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="resizeMode">The resize mode.</param>
        /// <returns>WriteableBitmap.</returns>
        internal static WriteableBitmap Resize(this WriteableBitmap source, int width, int height, Common.ResizeMode resizeMode = Common.ResizeMode.Stretch)
        {
            var rect = new Rect(0, 0, width, height);
            if (resizeMode == Common.ResizeMode.Crop)
            {
                float scaleX = (float)width / source.PixelWidth;
                float scaleY = (float)height / source.PixelHeight;
                var scaleFactor = Math.Max(scaleX, scaleY);
                int zoomWidth = (int)(source.PixelWidth * scaleFactor);
                int zoomHeight = (int)(source.PixelHeight * scaleFactor);
                int cropX = Math.Max((zoomWidth - width) / 2, 0);
                int cropY = Math.Max((zoomHeight - height) / 2, 0);
                rect = new Rect(-cropX, -cropY, zoomWidth, zoomHeight);
            }

            return Dispatcher.Invoke(() =>
            {
                var visual = new DrawingVisual();
                using (var context = visual.RenderOpen())
                {
                    context.DrawImage(source, rect);
                }

                var resizedBitmap = new RenderTargetBitmap(width, height, 96, 96, PixelFormats.Pbgra32);
                resizedBitmap.Render(visual);
                return new WriteableBitmap(resizedBitmap);
            });
        }


        /// <summary>
        /// Converts ImageTensor to WriteableBitmap.
        /// </summary>
        /// <param name="imageTensor">The image tensor.</param>
        /// <returns>WriteableBitmap.</returns>
        public static WriteableBitmap ToImage(this ImageTensor imageTensor)
        {
            var channels = imageTensor.Dimensions[1];
            var height = imageTensor.Dimensions[2];
            var width = imageTensor.Dimensions[3];
            return Dispatcher.Invoke(() =>
            {
                if (channels == 1)
                    return imageTensor.ToSingleChannelImage();

                var stride = width * 4;
                var pixelBuffer = new byte[height * stride];
                var writeableBitmap = new WriteableBitmap(width, height, 96, 96, PixelFormats.Pbgra32, null);
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
                return writeableBitmap;
            });
        }


        /// <summary>
        /// Converts WriteableBitmap to ImageTensor.
        /// </summary>
        /// <param name="writeableBitmap">The writeable bitmap.</param>
        /// <returns>ImageTensor.</returns>
        internal static ImageTensor ToTensor(this WriteableBitmap writeableBitmap)
        {
            var width = writeableBitmap.PixelWidth;
            var height = writeableBitmap.PixelHeight;
            var tensor = new ImageTensor(new[] { 1, 4, height, width });
            unsafe
            {
                byte* buffer = (byte*)writeableBitmap.BackBuffer.ToPointer();
                for (int y = 0; y < height; y++)
                {
                    for (int x = 0; x < width; x++)
                    {
                        int pixelIndex = (y * width + x) * 4; // BGRA
                        tensor[0, 0, y, x] = GetFloatValue(buffer[pixelIndex + 2]); // R
                        tensor[0, 1, y, x] = GetFloatValue(buffer[pixelIndex + 1]); // G
                        tensor[0, 2, y, x] = GetFloatValue(buffer[pixelIndex + 0]); // B
                        tensor[0, 3, y, x] = GetFloatValue(buffer[pixelIndex + 3]); // A
                    }
                }
            }
            return tensor;
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

            return writeableBitmap;
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
        /// Gets the rotation of an image file.
        /// </summary>
        /// <param name="imageUri">The image URI.</param>
        /// <returns>Rotation.</returns>
        private static Rotation GetRotation(Uri imageUri)
        {
            var bitmapFrame = BitmapFrame.Create(imageUri, BitmapCreateOptions.DelayCreation, BitmapCacheOption.None);
            if (bitmapFrame.Metadata is BitmapMetadata bitmapMetadata && bitmapMetadata.ContainsQuery(RotationQuery))
            {
                var queryResult = bitmapMetadata.GetQuery(RotationQuery);
                if (queryResult is ushort orientation)
                {
                    switch (orientation)
                    {
                        case 6:
                            return Rotation.Rotate90;
                        case 3:
                            return Rotation.Rotate180;
                        case 8:
                            return Rotation.Rotate270;
                    }
                }
            }
            return Rotation.Rotate0;
        }

    }
}
