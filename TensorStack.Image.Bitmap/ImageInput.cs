// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Drawing;
using TensorStack.Common;
using TensorStack.Common.Image;
using TensorStack.Common.Tensor;

namespace TensorStack.Image
{
    /// <summary>
    /// ImageInput implementation with System.Drawing.Bitmap.
    /// </summary>
    public class ImageInput : ImageInput<Bitmap>
    {
        private Bitmap _image;

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
        public ImageInput(Bitmap image)
            : base(image.ToTensor())
        {
            _image = image;
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="ImageInput"/> class.
        /// </summary>
        /// <param name="filename">The filename.</param>
        public ImageInput(string filename)
            : this(new Bitmap(filename)) { }


        /// <summary>
        /// Initializes a new instance of the <see cref="ImageInput"/> class.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="resizeMode">The resize mode.</param>
        public ImageInput(string filename, int width, int height, ResizeMode resizeMode = ResizeMode.Stretch) : this(new Bitmap(filename))
        {
            Resize(width, height, resizeMode);
        }


        /// <summary>
        /// Gets the image.
        /// </summary>
        public override Bitmap Image => _image;


        /// <summary>
        /// Saves the image.
        /// </summary>
        /// <param name="filename">The filename.</param>
        public override void Save(string filename)
        {
            _image.Save(filename);
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
            _image?.Dispose();
            _image = null;
            base.Dispose(disposing);
        }

    }
}
