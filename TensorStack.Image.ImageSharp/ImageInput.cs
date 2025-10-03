// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common.Tensor;

namespace TensorStack.Image
{
    /// <summary>
    /// ImageInput implementation with ImageSharp Image<Rgba32>.
    /// </summary>
    public class ImageInput : ImageInputBase
    {
        private readonly string _sourceFile;
        private Image<Rgba32> _image;

        /// <summary>
        /// Initializes a new instance of the <see cref="ImageInput"/> class.
        /// </summary>
        /// <param name="tensor">The tensor.</param>
        public ImageInput(ImageTensor tensor) : base(tensor)
        {
            _image = tensor.ToImageSharp();
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
            : this(Image<Rgba32>.Load<Rgba32>(filename))
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
        public ImageInput(string filename, int width, int height, Common.ResizeMode resizeMode = Common.ResizeMode.Stretch) : this(Image<Rgba32>.Load<Rgba32>(filename))
        {
            Resize(width, height, resizeMode);
        }


        /// <summary>
        /// Gets the image.
        /// </summary>
        public Image<Rgba32> Image => _image;

        /// <summary>
        /// Gets the source Image filename.
        /// </summary>
        public override string SourceFile => _sourceFile;


        /// <summary>
        /// Saves the specified image.
        /// </summary>
        public override void Save(string filename)
        {
            _image.SaveAsPng(filename);
        }


        /// <summary>
        /// Save the Image to file
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="cancellationToken">The cancellation token</param>
        public override async Task SaveAsync(string filename, CancellationToken cancellationToken = default)
        {
            await _image.SaveAsPngAsync(filename);
        }


        /// <summary>
        /// Called when Tensor data has changed
        /// </summary>
        protected override void OnTensorDataChanged()
        {
            base.OnTensorDataChanged();
            _image = this.ToImageSharp();
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
