﻿using System;
using System.IO;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;

namespace TensorStack.Image
{
    public static class ImageService
    {
        const string RotationQuery = "System.Photo.Orientation";
   
        /// <summary>
        /// Loads image from file.
        /// </summary>
        /// <param name="fileName">Name of the file.</param>
        /// <returns>BitmapSource.</returns>
        public static BitmapSource LoadFromFile(string fileName)
        {
            return Load(fileName);
        }


        /// <summary>
        /// Load as an asynchronous operation.
        /// </summary>
        /// <param name="filePath">The file path.</param>
        public static async Task<WriteableBitmap> LoadFromFileAsync(string filePath)
        {
            return await Task.Run(() => Load(filePath));
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
            var bitmapSource = new BitmapImage();
            bitmapSource.BeginInit();
            bitmapSource.Rotation = rotation;
            bitmapSource.UriSource = new Uri(filePath);
            bitmapSource.CacheOption = BitmapCacheOption.OnLoad;
            bitmapSource.EndInit();
            bitmapSource.Freeze();
            return bitmapSource.ToWriteableBitmap();
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
