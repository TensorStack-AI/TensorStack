// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using SixLabors.ImageSharp.Processing;
using TensorStack.Common.Image;
using TensorStack.Common.Tensor;

namespace TensorStack.Image
{
    /// <summary>
    /// ImageInput implementation with ImageSharp Image<Rgba32>.
    /// </summary>
    public class ImageInput : ImageInput<Image<Rgba32>>
    {
        private Image<Rgba32> _image;

        /// <summary>
        /// Initializes a new instance of the <see cref="ImageInput"/> class.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        public ImageInput(ImageTensor tensor) : base(tensor)
        {
            _image = tensor.ToImage();
        }


        /// <summary>
        /// Initializes a new instance of the <see cref="ImageInput"/> class.
        /// </summary>
        /// <param name="image">The image.</param>
        public ImageInput(Image<Rgba32> image)
            : base(image.ToTensor())
        {
            _image = image;
        }

        /// <summary>
        /// Initializes a new instance of the <see cref="ImageInput"/> class.
        /// </summary>
        /// <param name="filename">The filename.</param>
        public ImageInput(string filename)
            : this(Image<Rgba32>.Load<Rgba32>(filename)) { }


        /// <summary>
        /// Initializes a new instance of the <see cref="ImageInput"/> class.
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="resizeMode">The resize mode.</param>
        public ImageInput(string filename, int width, int height, Common.ResizeMode resizeMode = Common.ResizeMode.Stretch)
            : this(Image<Rgba32>.Load<Rgba32>(filename).Resize(width, height, resizeMode)) { }


        /// <summary>
        /// Gets the image.
        /// </summary>
        public override Image<Rgba32> Image => _image;


        /// <summary>
        /// Resizes the image and tensor.
        /// </summary>
        /// <param name="width">The width.</param>
        /// <param name="height">The height.</param>
        /// <param name="resizeMode">The resize mode.</param>
        public override void Resize(int width, int height, Common.ResizeMode resizeMode = Common.ResizeMode.Stretch)
        {
            _image.Mutate(x => x.Resize(width, height));
            UpdateTensor(_image.ToTensor());
        }


        /// <summary>
        /// Saves the specified image.
        /// </summary>
        /// <param name="filename">The filename.</param>
        public override void Save(string filename)
        {
            _image.SaveAsPng(filename);
        }


        /// <summary>
        /// Called when Tensor data has changed
        /// </summary>
        protected override void UpdateImage()
        {
            _image = this.ToImage();
            base.UpdateImage();
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
