// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Threading;
using System.Threading.Tasks;
using System.Windows.Media.Imaging;
using TensorStack.Common;
using TensorStack.Common.Tensor;

namespace TensorStack.Image
{
    /// <summary>
    /// ImageInput implementation with System.Windows.Media.Imaging.WriteableBitmap.
    /// </summary>
    public class ImageInput : ImageInputBase
    {
        private readonly string _sourceFile;
        private WriteableBitmap _image;

        /// <summary>
        /// Initializes a new instance of the <see cref="ImageInput"/> class.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        public ImageInput(ImageTensor tensor) : base(tensor)
        {
            _image = tensor.ToBitmapImage();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ImageInput"/> class.
        /// </summary>
        /// <param name="image">The image.</param>
        public ImageInput(WriteableBitmap image)
            : base(image.ToTensor())
        {
            _image = image;
            if (!_image.IsFrozen)
                _image.Freeze();
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ImageInput"/> class.
        /// </summary>
        /// <param name="image">The image.</param>
        public ImageInput(BitmapSource image)
            : this(new WriteableBitmap(image)) { }


        /// <summary>
        /// Initializes a new instance of the <see cref="ImageInput"/> class.
        /// </summary>
        /// <param name="filename">The filename.</param>
        public ImageInput(string filename)
            : this(ImageService.Load(filename))
        {
            _sourceFile = filename;
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="ImageInput"/> class.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="resizeMode">The resize mode.</param>
        public ImageInput(string filename, int width, int height, ResizeMode resizeMode = ResizeMode.Stretch)
            : this(ImageService.Load(filename))
        {
            Resize(width, height, resizeMode);
        }


        /// <summary>
        /// Gets the image.
        /// </summary>
        public WriteableBitmap Image => _image;

        /// <summary>
        /// Gets the source Image filename.
        /// </summary>
        public override string SourceFile => _sourceFile;


        /// <summary>
        /// Saves the image.
        /// </summary>
        /// <param name="filename">The filename.</param>
        public override void Save(string filename)
        {
            _image.Save(filename);
        }


        /// <summary>
        /// Save the Image to file
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="cancellationToken">The cancellation token</param>
        public override Task SaveAsync(string filename, CancellationToken cancellationToken = default)
        {
            return Task.Run(() => Save(filename), cancellationToken);
        }


        /// <summary>
        /// Called when Tensor data has changed
        /// </summary>
        protected override void OnTensorDataChanged()
        {
            base.OnTensorDataChanged();
            _image = this.ToBitmapImage();
        }


        /// <summary>
        /// Releases resources.
        /// </summary>
        protected override void Dispose(bool disposing)
        {
            _image = null;
            base.Dispose(disposing);
        }

    }
}
