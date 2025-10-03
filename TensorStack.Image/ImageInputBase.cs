// Copyright (c) TensorStack. All rights reserved.
// Licensed under the Apache 2.0 License.
using System.Threading;
using System.Threading.Tasks;
using TensorStack.Common.Tensor;

namespace TensorStack.Image
{
    public abstract class ImageInputBase : ImageTensor
    {
        /// <summary>
        /// Initializes a new instance of the <see cref="ImageInputBase"/> class.
        /// </summary>
        /// <param name="imageTensor">The Image tensor.</param>
        protected ImageInputBase(ImageTensor imageTensor)
            : base(imageTensor) { }

        /// <summary>
        /// Gets the source Image filename.
        /// </summary>
        public abstract string SourceFile { get; }

        /// <summary>
        /// Save the Image to file
        /// </summary>
        /// <param name="filename">The filename.</param>
        public abstract void Save(string filename);

        /// <summary>
        /// Save the Image to file
        /// </summary>
        /// <param name="filename">The filename.</param>
        /// <param name="cancellationToken">The cancellation token</param>
        public abstract Task SaveAsync(string filename, CancellationToken cancellationToken = default);
    }
}
